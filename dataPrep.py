import sys, pymesh, glob, os
import numpy as np

EXTENT = 100.0
CLEARANCE = 0.1

DATA_ROOT = 'ModelNet10'

def triangle_area(v1,v2,v3):
    return 0.5 * np.linalg.norm(np.cross(v2-v1, v3-v1),axis = 1)

def createGrid(mesh):
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

    # centering the model
    delta = (np.max(pt_cld,axis=0) + np.min(pt_cld,axis=0))/2.0
    if sum(np.abs(delta)) != 0.0:
        pt_cld -= delta

    # scaling the model to have largest extent of 100
    span = np.max(np.max(pt_cld,axis=0) - np.min(pt_cld,axis=0))
    pt_cld /= span
    pt_cld *= (EXTENT - 2*CLEARANCE)
    pt_cld += EXTENT/2.0


    matrices = [np.zeros((128,128,128)),
    np.zeros((64,64,64)),
    np.zeros((32,32,32)),
    np.zeros((16,16,16)),
    np.zeros((8,8,8)),
    np.zeros((4,4,4))]

    for D in range(len(matrices)):
        unit = EXTENT/float(2**(len(matrices)- D + 1))
        inds = (pt_cld/unit).astype(int)
        for i in inds:
            matrices[D][tuple(i)] += 1
        matrices[D] /= np.mean(matrices[D])

    return np.array(matrices)

if __name__ == '__main__':
    if len(sys.argv)>1:
        category = sys.argv[1]

        allFiles = glob.glob(DATA_ROOT + '/' + category + '/train/*.off')
        allFiles.extend(glob.glob(DATA_ROOT + '/' + category + '/test/*.off'))

        for model in allFiles:
            print(model)
            mesh = pymesh.load_mesh(model)
            pymesh.save_mesh(model[:-3] + 'stl', mesh)
            os.system('./voxelizer 126 10 ' + model[:-3] + 'stl ' + model[:-3] + 'vox')
            # os.system('./getConfigs ' + model[:-3] + 'vox ' + model[:-3] + 'txt')
            np.save(model[:-3] + 'npy',createGrid(mesh))
