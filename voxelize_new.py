import time, trimesh
import numpy as np

FILE = 'cube.stl'
DIM = 32
EXTENT = 100.0
CLEARANCE = 5.0
EXTRA_VOX = 1  # how many more to consider over the bounding region of triangles

coords = np.linspace(-(EXTENT + CLEARANCE)/2,(EXTENT + CLEARANCE)/2,DIM + 1)
unit = (EXTENT + CLEARANCE)/DIM     # side of a cube

# which cubes to consider for assigning configurations, and for initial in-out checks
probeMap = np.zeros((DIM,DIM,DIM),int)

# in-out map of corners of cubes, inside = 1
inOutGrid = np.zeros((DIM + 1,DIM + 1,DIM + 1))

def onSegment(p,q,r):
    bool1 = (q[:,0] <= np.max(np.concatenate([p[:,0].reshape((-1,1)),r[:,0].reshape((-1,1))],axis = 1),axis = 1))
    bool2 = q[:,0] >= np.min(np.concatenate([p[:,0].reshape((-1,1)),r[:,0].reshape((-1,1))],axis = 1),axis = 1)
    bool3 = q[:,1] <= np.max(np.concatenate([p[:,1].reshape((-1,1)),r[:,1].reshape((-1,1))],axis = 1),axis = 1)
    bool4 = q[:,1] <= np.min(np.concatenate([p[:,1].reshape((-1,1)),r[:,1].reshape((-1,1))],axis = 1),axis = 1)

    return bool1*bool2*bool3*bool4

def orientation(p,q,r):
    val = (q[:,1] - p[:,1])*(r[:,0] - q[:,0]) - (q[:,0] - p[:,0])*(r[:,1] - q[:,1])
    retVal = np.zeros((val.shape[0],1))
    retVal[np.where(val>0)] = 1
    retVal[np.where(val<0)] = 2
    return retVal

def areCornersIn(xy_inds,lsegs):
    p1 = np.array((xy_inds - DIM/2).astype(float)*unit)
    q1 = np.tile(np.array([EXTENT/2.0,EXTENT/2.0]),p1.shape[0]).reshape((-1,2))
    intersectionsSum = np.zeros((p1.shape[0],1))
    for lseg in lsegs:
        p2 = np.tile(lseg[0,0:2],p1.shape[0]).reshape((-1,2))
        q2 = np.tile(lseg[1,0:2],p1.shape[0]).reshape((-1,2))

        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        bool1 = (o1 != o2)* (o3 != o4)
        bool2 = (o1 == 0)* onSegment(p1, p2, q1).reshape((-1,1))
        bool3 = (o2 == 0)* onSegment(p1, q2, q1).reshape((-1,1))
        bool4 = (o3 == 0)* onSegment(p2, p1, q2).reshape((-1,1))
        bool5 = (o4 == 0)* onSegment(p2, q1, q2).reshape((-1,1))

        intersectionsSum += (bool1 + bool2 + bool3 + bool4 + bool5).astype(int)

    return np.remainder(intersectionsSum,2)

def safeInd(val):
    return max(DIM - 1,min(0,val))

orig_mesh = trimesh.load_mesh(FILE)
verts = np.array(orig_mesh.vertices)

# centering the model
delta = (np.max(orig_mesh.vertices,axis=0) + np.min(orig_mesh.vertices,axis=0))/2.0
if sum(np.abs(delta)) != 0.0:
    verts -= delta

# scaling the model to have largest extent of 100
span = np.max(np.max(verts,axis=0) - np.min(verts,axis=0))
verts /= span
verts *= EXTENT - CLEARANCE

# the mesh to be used
mesh = trimesh.base.Trimesh(verts,orig_mesh.faces)

start_time = time.time()


# find interior cube corners for every layer
z_ind = 0
for z in coords:
    plane_normal = (0,0,1)
    plane_origin = (0,0,z)

    cut_slice = trimesh.intersections.mesh_plane(mesh,plane_normal,plane_origin)
    cut_slice = np.array(cut_slice)[:,:,0:2] # [n x 2 x 3]; n - no. of line segments
    if cut_slice.shape[0] == 0:
        z_ind += 1
        continue

    voxs = (cut_slice/unit + DIM/2).astype(int)

    # for each line segment, collect corners to be tested [those within bounding box of segment]
    xy = None
    for i in range(voxs.shape[0]):
        '''
        Here, x and y refer to indices, not actual coordinates.
        So, xy contains list of indices of the cube corners which must be checked
        '''
        lseg = voxs[i,:,:]

        xmin, ymin = np.clip(np.min(lseg,axis = 0) - EXTRA_VOX,0,DIM)
        xmax, ymax = np.clip(np.max(lseg,axis = 0) + EXTRA_VOX,0,DIM)

        probeMap[xmin:xmax, ymin:ymax, z_ind] = 1
        probeMap[xmin:xmax, ymin:ymax, safeInd(z_ind - 1)] = 1
        probeMap[xmin:xmax, ymin:ymax, safeInd(z_ind + 1)] = 1

        xtot, ytot = xmax - xmin + 1, ymax - ymin + 1

        xs = np.linspace(xmin, xmax, xtot)
        ys = np.linspace(ymin, ymax, ytot)

        xy_ = np.concatenate([np.repeat(xs,ytot).reshape((-1,1)),
                            np.tile(ys,xtot).reshape((-1,1))],axis = 1)

        if i == 0:
            xy = xy_
        else:
            xy = np.concatenate([xy,xy_], axis = 0)

    xs = np.linspace(0, DIM, DIM + 1)
    xy = np.concatenate([np.repeat(xs,DIM + 1).reshape((-1,1)),
                        np.tile(xs,DIM + 1).reshape((-1,1))],axis = 1)

    xyInOut = areCornersIn(xy,cut_slice)
    non_zeros = xyInOut.nonzero()
    # print((xy[non_zeros[0]] - DIM/2.0)*unit)

    xyInsides = xy[non_zeros[0]].astype(int)
    inOutGrid[xyInsides[:,0],xyInsides[:,1],np.repeat(z_ind,xyInsides.shape[0])] = 1

    '''
    TO-DO: Seed fill!
    '''

    z_ind += 1

x_ind = 0
for x in coords:
    plane_normal = (1,0,0)
    plane_origin = (x,0,0)

    cut_slice = trimesh.intersections.mesh_plane(mesh,plane_normal,plane_origin)
    cut_slice = np.array(cut_slice)[:,:,1:3] # [n x 2 x 3]; n - no. of line segments
    if cut_slice.shape[0] == 0:
        x_ind += 1
        continue

    voxs = (cut_slice/unit + DIM/2).astype(int)

    # for each line segment, collect corners to be tested [those within bounding box of segment]
    yz = None
    for i in range(voxs.shape[0]):
        lseg = voxs[i,:,:]

        ymin, zmin = np.clip(np.min(lseg,axis = 0) - EXTRA_VOX,0,DIM)
        ymax, zmax = np.clip(np.max(lseg,axis = 0) + EXTRA_VOX,0,DIM)

        probeMap[x_ind, ymin:ymax, zmin:zmax] = 1
        probeMap[safeInd(x_ind - 1), ymin:ymax, zmin:zmax] = 1
        probeMap[safeInd(x_ind + 1), ymin:ymax, zmin:zmax] = 1

        ytot, ztot = ymax - ymin + 1, zmax - zmin + 1

        ys = np.linspace(ymin, ymax, ytot)
        zs = np.linspace(zmin, zmax, ztot)

        yz_ = np.concatenate([np.repeat(ys,ztot).reshape((-1,1)),
                            np.tile(zs,ytot).reshape((-1,1))],axis = 1)

        if i == 0:
            yz = yz_
        else:
            yz = np.concatenate([yz,yz_], axis = 0)

    yzInOut = areCornersIn(yz,cut_slice)
    non_zeros = yzInOut.nonzero()

    yzInsides = yz[non_zeros[0]].astype(int)
    inOutGrid[np.repeat(x_ind,yzInsides.shape[0]),yzInsides[:,0],yzInsides[:,1]] = 1

    '''
    TO-DO: Seed fill!
    '''

    x_ind += 1

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


powers = np.tile(np.array([2,1,32,16,4,8,64,128]),DIM**3)
zero1 = np.array([0,1])
addend = np.array([np.repeat(zero1,4),np.tile(np.repeat(zero1,2),2),np.tile(zero1,4)])
inds = np.linspace(0,DIM - 1,DIM).astype(int)
toBeProbed = np.concatenate(  [np.repeat(inds,DIM**2).reshape(-1,1),
                        np.tile(np.repeat(inds,DIM),DIM).reshape(-1,1),
                        np.tile(inds,DIM**2).reshape(-1,1)],   axis = 1)
temp_ = np.repeat(toBeProbed,8, axis = 0)
addend = np.tile(addend.transpose(),(DIM**3,1))
indices = (temp_ + addend).astype(int)
val = np.sum((inOutGrid[indices[:,0],indices[:,1],indices[:,2]]*powers).reshape(-1,8), axis = 1).astype(int)



triPts = np.empty((DIM,DIM,DIM,12,3))
u2 = unit/2.0
'''
values to be added to base corner of each cube,
to get the vertices which will form the triangles of the mesh
'''
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
for x in range(DIM):
    for y in range(DIM):
        for z in range(DIM):
            triPts[x,y,z] = np.tile(
                            np.array([coords[x],coords[y],coords[z]]).reshape(1,3),
                            (12,1)) + triAddend

vx = inOutGrid.nonzero()[0]
vy = inOutGrid.nonzero()[1]
vz = inOutGrid.nonzero()[2]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# vx = val.nonzero()[0]
# ax.scatter(toBeProbed[vx,0],toBeProbed[vx,1],-toBeProbed[vx,2], zdir='z', c= 'red')

ax.scatter(vx,vy,-vz, zdir='z', c= 'red')
# ax.voxels(inOutGrid)
plt.show()


vnz = val.nonzero()[0]
labels = val[vnz]
xyzs = toBeProbed[vnz,:]

cfgs = np.load('configs.npy')

with open('test.stl','w') as stlFile:
    stlFile.write('solid model\n')
    for xyz, label in zip(xyzs, labels):
        if label == 0 | label == 255:
            continue
        for triAng in cfgs[label]:
            v1 = triPts[xyz[0],xyz[1],xyz[2],triAng[0],:]
            v2 = triPts[xyz[0],xyz[1],xyz[2],triAng[1],:]
            v3 = triPts[xyz[0],xyz[1],xyz[2],triAng[2],:]
            stlFile.write('\tfacet normal 0 0 1\n')
            stlFile.write('\t\touter loop\n')
            stlFile.write('\t\t\tvertex ' + str(v1[0]) + ' ' + str(v1[1]) + ' ' + str(v1[2]) + '\n ')
            stlFile.write('\t\t\tvertex ' + str(v2[0]) + ' ' + str(v2[1]) + ' ' + str(v2[2]) + '\n ')
            stlFile.write('\t\t\tvertex ' + str(v3[0]) + ' ' + str(v3[1]) + ' ' + str(v3[2]) + '\n ')
            stlFile.write('\t\tendloop\n')
            stlFile.write('\tendfacet\n')
    stlFile.write('endsolid model\n')




print('It took ' + str(time.time() - start_time) + ' seconds.')
