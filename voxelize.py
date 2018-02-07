import pymesh, time
import numpy as np

FILE = 'model.obj'
DIM = 64

def addvox(Q,center):
    for i in [-1,0,1]:
        for j in [-1,0,1]:
            for k in [-1,0,1]:
                Q.append(center + [i,j,k])

def maxdist(v1,v2,v3):
    d1 = np.sqrt(sum((v1 - v2)**2))
    d2 = np.sqrt(sum((v1 - v3)**2))
    d3 = np.sqrt(sum((v2 - v3)**2))
    return max(d1,d2,d3)

orig_mesh = pymesh.load_mesh(FILE)
verts = orig_mesh.vertices

delta = np.max(verts,axis=0) + np.min(verts,axis=0)
if sum(delta) != 0.0:
    verts -= delta  # centering the model

span = np.max(np.max(verts,axis=0) - np.min(verts,axis=0))
# scaling the model to have largest extent of 100
verts /= span
verts *= 100

voxels = np.zeros((DIM,DIM,DIM))
coords = np.linspace(-50,50,DIM*2)
unit = 100.0/DIM

probeMap = np.zeros((DIM,DIM,DIM),int)

count = 0
start_time = time.time()
for face in orig_mesh.faces:

    # Q = []
    # centroid = (verts[face[0]] + verts[face[1]] + verts[face[2]])/3
    numSteps = int(maxdist(verts[face[0]],verts[face[1]],verts[face[2]])/(0.25*unit))
    u = np.repeat(np.linspace(0,1,numSteps),numSteps)
    v = np.tile(np.linspace(0,1,numSteps),numSteps)
    exceeds = np.where((u + v)>1)[0]
    u[exceeds] = 1 - u[exceeds]
    v[exceeds] = 1 - v[exceeds]
    w = 1 - (u + v)

    # uvw.shape : [numSteps,3]
    uvw = np.concatenate((u.reshape((-1,1)),v.reshape((-1,1)),w.reshape((-1,1))),axis =1)
    triangle = np.array([verts[face[0]],verts[face[1]],verts[face[2]]])
    voxes = (np.matmul(uvw,triangle)/unit + DIM/2 - 1).astype(int)

    probeMap[voxes[:,0],voxes[:,1],voxes[:,2]] = 1

    # for u in :
    #     for v in :
    #         temp = u + v
    #         if temp>1:
    #             u = 1 - u
    #             v = 1 - v
    #         w = 1 - (u + v)
    #         pt = u*verts[face[0]] + v*verts[face[1]] + w*verts[face[2]]
    #         vox = (pt/unit + DIM/2).astype(int)
    #         if probeMap[tuple(vox)] == 0:
    #             # print(vox)
    #             probeMap[tuple(vox)] = 1
    ''' TO-DO
    - compute vector originating from centroid and normal to traingle
    - compute vectors to 4 corners of voxels
    - dot product
    '''

    # core_vox = (centroid/unit+512).astype(int)
    # # adding the core and surrounding voxels
    # addvox(Q,core_vox)
    count += 1
    # print(count)
    # break

print('It took ' + str(time.time() - start_time) + ' seconds.')
print(np.sum(probeMap))
# x,y,z = probeMap.nonzero()
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, -z, zdir='z', c= 'red')
# plt.savefig("demo3.png")
