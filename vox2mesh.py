import time, trimesh, os, pymesh, re
import numpy as np

DIM = 256
EXTENT = 100.0
CLEARANCE = 2.0 # space between the model and the bounds of grid

coords = np.linspace(-EXTENT/2,EXTENT/2,DIM + 1)
unit = EXTENT/DIM     # side of a cube

# in-out map of corners of cubes, inside = 1
inOutGrid = np.zeros((DIM + 1,DIM + 1,DIM + 1))

start_time = time.time()

# with open('bathtub_0001.vox','rb') as f:
#     allVals = f.read()
#
# inCorners = np.array(allVals.replace('\n',' ').split(' ')[5:-1]).reshape(-1,3).astype(int)
# inCorners += np.array([3, 3, 3])
#
# inOutGrid[inCorners[:,0],inCorners[:,1],inCorners[:,2]] = 1

# toBeProbed = np.array(probeMap.nonzero()).transpose()
# # order of powers of 2 based on matching with the convention used in the online
# # configurations sheet
# powers = np.tile(np.array([2,1,32,16,4,8,64,128]),toBeProbed.shape[0])
# zero1 = np.array([0,1])
# addend = np.array([np.repeat(zero1,4),np.tile(np.repeat(zero1,2),2),np.tile(zero1,4)])
# temp_ = np.repeat(toBeProbed,8, axis = 0)
# addend = np.tile(addend.transpose(),(toBeProbed.shape[0],1))
# indices = (temp_ + addend).astype(int)
# val = np.sum((inOutGrid[indices[:,0],indices[:,1],indices[:,2]]*powers).reshape(-1,8), axis = 1).astype(int)


# powers = np.tile(np.array([2,1,32,16,4,8,64,128]),DIM**3)
# zero1 = np.array([0,1])
# addend = np.array([np.repeat(zero1,4),np.tile(np.repeat(zero1,2),2),np.tile(zero1,4)])
# inds = np.linspace(0,DIM - 1,DIM).astype(int)
# toBeProbed = np.concatenate(  [np.repeat(inds,DIM**2).reshape(-1,1),
#                         np.tile(np.repeat(inds,DIM),DIM).reshape(-1,1),
#                         np.tile(inds,DIM**2).reshape(-1,1)],   axis = 1)
# temp_ = np.repeat(toBeProbed,8, axis = 0)
# addend = np.tile(addend.transpose(),(DIM**3,1))
# indices = (temp_ + addend).astype(int)
# val = np.sum((inOutGrid[indices[:,0],indices[:,1],indices[:,2]]*powers).reshape(-1,8), axis = 1).astype(int)


zero1 = np.array([0,1])
vals = np.zeros((DIM,DIM,DIM))
powers2 = np.array([2,1,32,16,4,8,64,128])
addend2 = np.array([np.repeat(zero1,4),np.tile(np.repeat(zero1,2),2),np.tile(zero1,4)])

x = 0
while x<DIM:
    y = 0
    while y<DIM:
        z = 0
        while z<DIM:
            temp2 = np.repeat(np.array([x,y,z]),8).reshape((3,8))
            indices_ = temp2 + addend2
            vals[x,y,z] = (int(np.sum(inOutGrid[indices_[0,:],indices_[1,:],indices_[2,:]]*powers2)))
            z += 1
        y += 1
    x += 1

vals = vals.astype(int)

#
# if os.path.exists('triPts' + str(DIM) + '.npy') == True:
#     triPts = np.load('triPts' + str(DIM) + '.npy')
# else:
#     triPts = np.empty((DIM,DIM,DIM,12,3))
#     u2 = unit/2.0
#     '''
#     values to be added to base corner of each cube,
#     to get the vertices which will form the triangles of the mesh
#     '''
#     triAddend = np.array([[0,0,u2],
#                         [u2,0,0],
#                         [unit,0,u2],
#                         [u2,0,unit],
#                         [0,unit,u2],
#                         [u2,unit,0],
#                         [unit,unit,u2],
#                         [u2,unit,unit],
#                         [0,u2,unit],
#                         [0,u2,0],
#                         [unit,u2,0],
#                         [unit,u2,unit]])
#     for x in range(DIM):
#         for y in range(DIM):
#             for z in range(DIM):
#                 triPts[x,y,z] = np.tile(
#                                 np.array([coords[x],coords[y],coords[z]]).reshape(1,3),
#                                 (12,1)) + triAddend
#     np.save('triPts' + str(DIM) + '.npy',triPts)



# vx = inOutGrid.nonzero()[0]
# vy = inOutGrid.nonzero()[1]
# vz = inOutGrid.nonzero()[2]
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # vx = val.nonzero()[0]
# # ax.scatter(toBeProbed[vx,0],toBeProbed[vx,1],-toBeProbed[vx,2], zdir='z', c= 'red')
#
# ax.scatter(vx,vy,-vz, zdir='z', c= 'red')
# # ax.voxels(inOutGrid)
# plt.show()
u2 = unit/2.0
triAddend = np.array([[0,0,u2],
                    [u2,0,0],
                    [unit,0,u2],
                    [u2,0,unit],
                    [0,unit,u2],
                    [u2,unit,0],
                    [unit,unit,u2],
                    [u2,unit,unit],
                    [0,u2,unit],
                    [0,u2,0],
                    [unit,u2,0],
                    [unit,u2,unit]])

with open('temp.txt','rb') as f:
    nzValsString = f.read()

nzVals = np.array(nzValsString.replace('\n',' ').split(' ')[:-1]).astype(int).reshape(-1,4)

labels = nzVals[:,3]
xyzs = nzVals[:,0:3]

cfgs = np.load('configsNew.npy')


with open('test.stl','w') as stlFile:
    stlFile.write('solid model\n')
    for xyz, label in zip(xyzs, labels):
        if label == 0 | label == 255:
            continue
        for triAng in np.array(cfgs[label]).reshape(-1,3):
            basePt = np.array([coords[xyz[0]],coords[xyz[1]],coords[xyz[2]]]).reshape(1,3)
            v1 = basePt + triAddend[triAng[0]]
            v2 = basePt + triAddend[triAng[1]]
            v3 = basePt + triAddend[triAng[2]]
            n = np.cross(v2 - v1, v3 - v1)
            stlFile.write('\tfacet normal ' + str(n[0]) + ' ' + str(n[1]) + ' ' + str(n[2]) + '\n ')
            stlFile.write('\t\touter loop\n')
            stlFile.write('\t\t\tvertex ' + str(v1[0]) + ' ' + str(v1[1]) + ' ' + str(v1[2]) + '\n ')
            stlFile.write('\t\t\tvertex ' + str(v2[0]) + ' ' + str(v2[1]) + ' ' + str(v2[2]) + '\n ')
            stlFile.write('\t\t\tvertex ' + str(v3[0]) + ' ' + str(v3[1]) + ' ' + str(v3[2]) + '\n ')
            stlFile.write('\t\tendloop\n')
            stlFile.write('\tendfacet\n')
    stlFile.write('endsolid model\n')


print('It took ' + str(time.time() - start_time) + ' seconds.')
