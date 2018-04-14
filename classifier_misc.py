from enum import Enum
import numpy as np
import trimesh

EXTENT = 100.0
CLEARANCE = 0.1
L2PARAM = 0.02
LEARNING_RATE = 1e-5
B1 = 0.9
B2 = 0.9

BATCH_TRAIN = 64
BATCH_EVAL = 64
NUM_ITERS = 40000

eval_start_delay = 600
eval_delay = 600

REG_PARAM = 0.05

TRAIN_FILE_LIST = 'trainFilesShuffled.txt'
EVAL_FILE_LIST = 'evalFiles.txt'
NUM_CLASSES = 10
IMG_RES = 256

with open(EVAL_FILE_LIST,'r') as f:
        evalFiles = np.array(f.read().split('\n')[:-1])
NUM_FILES_EVAL = evalFiles.shape[0]

CAT_DICT = {'bathtub':0,
            'bed':1,
            'chair':2,
            'desk':3,
            'dresser':4,
            'monitor':5,
            'night_stand':6,
            'sofa':7,
            'table':8,
            'toilet':9}

CATEGORIES = ['bathtub','bed','chair','desk','dresser','monitor','night_stand','sofa','table','toilet']

class axis(Enum):
    x = 0
    y = 1
    z = 2

def triangle_area(v1,v2,v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1),axis = 1)

def get_image(mesh, img_axis, random_num_pts = False, n = 2**14, use_seed = False):
    img_res = IMG_RES
    if use_seed:
        np.random.seed(2)
    faces = mesh.faces
    v1s,v2s,v3s = mesh.vertices[faces[:,0],:],mesh.vertices[faces[:,1],:],mesh.vertices[faces[:,2],:]

    areas = triangle_area(v1s,v2s,v3s)

    probs = areas/areas.sum()
    # number of points to sample
    if random_num_pts == True:
        n = np.random.randint(2**10,2**14)
    weighted_rand_inds = np.random.choice(range(len(areas)),size = n, p = probs)

    sel_v1s = v1s[weighted_rand_inds]
    sel_v2s = v2s[weighted_rand_inds]
    sel_v3s = v3s[weighted_rand_inds]

    # barycentric co-ords
    u = np.random.rand(n,1)
    v = np.random.rand(n,1)

    invalids = u + v >1 # u+v+w =1

    u[invalids] = 1 - u[invalids]
    v[invalids] = 1 - v[invalids]
    w = 1-(u+v)

    pt_cld = (sel_v1s * u) + (sel_v2s * v) + (sel_v3s * w)   

    # centering the model
    delta = (np.max(pt_cld,axis=0) + np.min(pt_cld,axis=0))/2.0
    if sum(np.abs(delta)) != 0.0:
        pt_cld -= delta

    # scaling the model to have largest extent of 100
    span = np.max(np.max(pt_cld,axis=0) - np.min(pt_cld,axis=0))
    pt_cld /= span
    pt_cld *= (EXTENT - 2*CLEARANCE)
    pt_cld += EXTENT/2.0    # to get indices correctly on dividing by unit length 

    # getting the image 
    unit = EXTENT/float(img_res)

    img = np.zeros((img_res,img_res))    
    inds_xyz = (pt_cld/unit).astype(int)

    # inds_img = inds_xyz[np.where(inds_xyz[:,img_axis.value] == int(img_res/2))[0]]
    # for i in range(20):
    #     inds_img = np.concatenate([inds_img, inds_xyz[np.where(inds_xyz[:,img_axis.value] == int(img_res/2 - i))[0]] ],axis = 0)
    #     inds_img = np.concatenate([inds_img, inds_xyz[np.where(inds_xyz[:,img_axis.value] == int(img_res/2 + i))[0]] ],axis = 0)
    
    img_axes_map = {0:[2,1],1:[2,0],2:[0,1]}
    plane_axes = img_axes_map[img_axis.value]    

    '''
    # binary values    
    img[inds_img[:,plane_axes[0]], inds_img[:,plane_axes[1]]] = 1
    '''
    # if density to be encoded
    # for pix in inds_img:
    #     img[pix[img_axes_map[img_axis.value][0]], pix[img_axes_map[img_axis.value][1]]] = 1

    level = 20
    level_pixel_map = {6:0.3, 5:0.4, 4:0.5, 3:0.6, 2:0.7, 1:0.8, 0:0.9}
    while level>0:
        inds_img = inds_xyz[np.where(inds_xyz[:,img_axis.value] == (int(img_res/2) - level) )[0]]
        img[inds_img[:,plane_axes[0]], inds_img[:,plane_axes[1]]] = level_pixel_map[int(level/3)]
        inds_img = inds_xyz[np.where(inds_xyz[:,img_axis.value] == (int(img_res/2) + level) )[0]]
        img[inds_img[:,plane_axes[0]], inds_img[:,plane_axes[1]]] = level_pixel_map[int(level/3)]
        level -= 1

    inds_img = inds_xyz[np.where(inds_xyz[:,img_axis.value] == int(img_res/2))[0]]
    img[inds_img[:,plane_axes[0]], inds_img[:,plane_axes[1]]] = 1
    # np.clip(img,0,1)
    # nz = np.where(img !=0 )
    # img[nz[0], nz[1]] *= np.mean(img)
    # max_intensity = np.max(img)
    # if  max_intensity != 0:
    #     img /= max_intensity
    

    return img

def get_images(mesh, img_axis, **kwargs):
    angles = [30,90,120,150]
    imgs = np.empty((IMG_RES, IMG_RES, len(angles)+1))
    imgs[:,:,0] = get_image(mesh, img_axis, **kwargs)
    
    axis_vector_map = {axis.x:[0,1,0], axis.y:[0,0,1], axis.z:[1,0,0]}
    for i in range(len(angles)):
        angle = angles[i]
        # rotate about given axis  
        angle_rad = angle*np.pi/180.0      
        M = trimesh.transformations.rotation_matrix(angle_rad,axis_vector_map[img_axis])
        mesh.apply_transform(M)
        imgs[:,:,i+1] = get_image(mesh, img_axis, **kwargs)  
    return imgs
