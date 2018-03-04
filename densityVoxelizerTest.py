import numpy as np
import pymesh, time

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

EXTENT = 100.0
CLEARANCE = 0.1

'''
Start with 1024 x 1024 x 1024 dimension
@ next level, build 512 x 512 x 512 by using the previous one, and so on... till 2 x 2 x 2

or Try the below one for each unit size. See which one's faster.

Find the unit size.
for each point, divide the x, y and z by the unit. That is the corresponding block where it belongs.
'''
def read_pts(fname):
    with open(fname,'r') as myfile:
        data = myfile.readlines()
    return np.loadtxt(data)

def triangle_area(v1,v2,v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1),axis = 1)

def createGrid(fname):
    t0 = time.time()
    print(fname)
    mesh = pymesh.load_mesh(fname)
    faces = mesh.faces
    v1s,v2s,v3s = mesh.vertices[faces[:,0],:],mesh.vertices[faces[:,1],:],mesh.vertices[faces[:,2],:]

    areas = triangle_area(v1s,v2s,v3s)

    probs = areas/areas.sum()
    n = 2**14
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
    print(str(time.time() - t0) + ' seconds.')
    t0 = time.time()

    # centering the model
    delta = (np.max(pt_cld,axis=0) + np.min(pt_cld,axis=0))/2.0
    if sum(np.abs(delta)) != 0.0:
        pt_cld -= delta

    # scaling the model to have largest extent of 100
    span = np.max(np.max(pt_cld,axis=0) - np.min(pt_cld,axis=0))
    pt_cld /= span
    pt_cld *= (EXTENT - 2*CLEARANCE)
    pt_cld += EXTENT/2.0


    matrices1 = [np.zeros((512,512,512)),
    np.zeros((256,256,256)),
    np.zeros((128,128,128)),
    np.zeros((64,64,64)),
    np.zeros((32,32,32)),
    np.zeros((16,16,16)),
    np.zeros((8,8,8)),
    np.zeros((4,4,4))]

    for D in range(8):
        unit = EXTENT/float(2**(9-D))
        inds = (pt_cld/unit).astype(int)
        for i in inds:
            matrices1[D][tuple(i)] += 1

    # D = np.arange(8)
    # unit = EXTENT/np.array([512,256,128,64,32,16,8,4]).astype(float)
    # inds = np.zeros((8,pt_cld.shape[0],pt_cld.shape[1]))
    # for i in range(8):
    #     inds[i,:,:] = (pt_cld/unit[i]).astype(int)
    # for i in range(pt_cld.shape[0]):
    #         matrices1[0][tuple(inds[0,i,:].astype(int))] += 1
    #         matrices1[1][tuple(inds[1,i,:].astype(int))] += 1
    #         matrices1[2][tuple(inds[2,i,:].astype(int))] += 1
    #         matrices1[3][tuple(inds[3,i,:].astype(int))] += 1
    #         matrices1[4][tuple(inds[4,i,:].astype(int))] += 1
    #         matrices1[5][tuple(inds[5,i,:].astype(int))] += 1
    #         matrices1[6][tuple(inds[6,i,:].astype(int))] += 1
    #         matrices1[7][tuple(inds[7,i,:].astype(int))] += 1

    print(str(time.time() - t0) + ' seconds.')

    # matrices = [np.zeros((512,512,512)),
    # np.zeros((256,256,256)),
    # np.zeros((128,128,128)),
    # np.zeros((64,64,64)),
    # np.zeros((32,32,32)),
    # np.zeros((16,16,16)),
    # np.zeros((8,8,8)),
    # np.zeros((4,4,4))]
    #
    # unit = EXTENT/float(512)
    # inds = (pt_cld/unit).astype(int)
    # for i in inds:
    #     matrices[0][tuple(i)] += 1
    #
    # for D in range(2,8):
    #     DIM = 2**(9-D)
    #     inds1 = np.arange(DIM).astype(int)
    #     inds2 = np.arange(DIM).astype(int)  * 2
    #     zero1 = np.array([0,1])
    #
    #     indexAddend = np.array([[0,0,0],
    #     [0,0,1],
    #     [0,1,0],
    #     [0,1,1],
    #     [1,0,0],
    #     [1,0,1],
    #     [1,1,0],
    #     [1,1,1]])
    #
    #     ix = np.repeat(inds2,DIM**2).reshape(-1,1)
    #     iy = np.tile(np.repeat(inds2,DIM),DIM).reshape(-1,1)
    #     iz = np.tile(inds2,DIM**2).reshape(-1,1)
    #     ixyz = np.tile(np.concatenate((ix,iy,iz),axis = 1),(8,1)) + np.repeat(indexAddend,DIM**3,axis = 0)
    #     matrices[D][inds1,inds1,inds1] = np.sum((matrices[D - 1][ixyz[:,0],ixyz[:,1],ixyz[:,2]]).reshape(-1,8),axis = 1)
    #     # inds2 = np.arange(DIM).astype(int)
    #     # inds2_ = np.arange(DIM).astype(int) * 2
    #     # matrices[D][inds2,inds2,inds2] = matrices[D - 1][inds2_,inds2_,inds2_] +\
    #     # matrices[D - 1][inds2_,inds2_,inds2_ + 1] +\
    #     # matrices[D - 1][inds2_,inds2_ + 1,inds2_] +\
    #     # matrices[D - 1][inds2_,inds2_ + 1,inds2_ + 1] +\
    #     # matrices[D - 1][inds2_ + 1,inds2_,inds2_] +\
    #     # matrices[D - 1][inds2_ + 1,inds2_,inds2_ + 1] +\
    #     # matrices[D - 1][inds2_ + 1,inds2_ + 1,inds2_] +\
    #     # matrices[D - 1][inds2_ + 1,inds2_ + 1,inds2_ + 1]

    # return matrices1,matrices


createGrid('sample.off')


# inds = pc.nonzero()
#
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# ax.scatter(inds[0],inds[1],inds[2],color='b',marker='.')
# # ax.scatter(pc[:,0],pc[:,1],pc[:,2],color='b',marker='.')
# plt.show()
