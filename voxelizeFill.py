import time, trimesh
import numpy as np

FILE = 'ModelNet10/bathtub/train/bathtub_0002.off'
DIM = 64
EXTENT = 100.0
CLEARANCE = 5.0 # space between the model and the bounds of grid

coords = np.linspace(-EXTENT/2,EXTENT/2,DIM + 1)
unit = EXTENT/DIM     # side of a cube

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

def getVoxInd(coord):
    return ((coord + EXTENT/2)*DIM/EXTENT - 1).astype(int)

def findLeftRight(lseg):
    x1, x2, y1, y2 = lseg[0,0],lseg[1,0],lseg[0,1],lseg[1,1]
    x, y = x2 - x1, y2 - y1
    if (abs(y) > abs(x)):
        return np.array([1,0])
    else:
        return np.array([0,1])

def sign(a):
    if a<0:
        return -1
    elif a>0:
        return 1
    return 0

def setBoundaryPixels(lseg,vz):
    x1, x2, y1, y2 = lseg[0,0],lseg[1,0],lseg[0,1],lseg[1,1]

    dx, dy, x, y = abs(x2 - x1), abs(y2 - y1), x1, y1
    S1, S2 = sign(x2 - x1), sign(y2 - y1)

    if dy>dx:
        t = dx
        dx = dy
        dy = dx
        interchanged = 1
    else:
        interchanged = 0

    E, A, B = 2 * dy - dx, 2 * dy, 2 * dy - 2 * dx

    for i in range(dx):
        if E<0:
            if interchanged == 1:
                y += S2
            else:
                x += S1
            E += A
        else:
            y += S2
            x += S1
            E += B
        inOutGrid[x,y,vz] = 1
        inOutGrid[x + 1,y,vz] = 1
        inOutGrid[x,y + 1,vz] = 1
        inOutGrid[x + 1,y + 1,vz] = 1

def boundaryFill(seedx,seedy,z):
    Q = [(seedx,seedy)]
    while len(Q)>0:
        seed = Q[0]
        # print(len(Q))
        Q.remove(Q[0])
        # print(len(Q))
        # print(seed)
        if min(seed) < 0 | max(seed) > DIM:
            # print('overflow')
            continue
        # if (inOutGrid[seed[0],seed[1],z] == 1):
            # print('One!')
        if (inOutGrid[seed[0],seed[1],z] == 0):
            inOutGrid[seed[0],seed[1],z] = 1
            if min(seed) > 1 | max(seed) < DIM - 1:
                if (inOutGrid[seed[0] - 1,seed[1],z] == 0):
                    Q.append((seed[0] - 1,seed[1]))
                if (inOutGrid[seed[0] + 1,seed[1],z] == 0):
                    Q.append((seed[0] + 1,seed[1]))
                if (inOutGrid[seed[0],seed[1] - 1,z] == 0):
                    Q.append((seed[0],seed[1] - 1))
                if (inOutGrid[seed[0],seed[1] + 1,z] == 0):
                    Q.append((seed[0],seed[1] + 1))


orig_mesh = trimesh.load_mesh(FILE)
verts = np.array(orig_mesh.vertices)

# centering the model
delta = (np.max(orig_mesh.vertices,axis=0) + np.min(orig_mesh.vertices,axis=0))/2.0
if sum(np.abs(delta)) != 0.0:
    verts -= delta

# scaling the model to have largest extent of 100
span = np.max(np.max(verts,axis=0) - np.min(verts,axis=0))
verts /= span
verts *= (EXTENT - 2*CLEARANCE)

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

    # line segments pf the polygon, in the form of indices of the cubes representing the end points
    voxs = getVoxInd(cut_slice)

    prospectiveSeeds = []

    # for each line segment, collect corners to be tested [those within bounding box of segment]
    for i in range(voxs.shape[0]):
        '''
        Here, x and y refer to indices, not actual coordinates.
        So, xy contains list of indices of the cube corners which must be checked
        '''
        lseg = voxs[i,:,:]
        mids = (lseg[0,:] + lseg[1,:])/2

        vec = findLeftRight(lseg)
        prospectiveSeeds.append(mids + vec)
        prospectiveSeeds.append(mids - vec)
        prospectiveSeeds.append(mids + 2*vec)
        prospectiveSeeds.append(mids - 2*vec)
        prospectiveSeeds.append(mids + 3*vec)
        prospectiveSeeds.append(mids - 3*vec)

        setBoundaryPixels(lseg,z_ind)

    prospectiveSeeds = np.array(prospectiveSeeds)
    seedsInOut = areCornersIn(prospectiveSeeds,cut_slice)
    non_zeros = seedsInOut.nonzero()

    seeds = prospectiveSeeds[non_zeros[0]].astype(int)

    print(z_ind)
    # boundaryFill(0,0, z_ind)
    # for seed in seeds:
    #     print(seed)
    #     if inOutGrid[seed[0],seed[1], z_ind] == 0:
    #         print(seed)
    #         boundaryFill(seed[0],seed[1], z_ind)



    z_ind += 1


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
