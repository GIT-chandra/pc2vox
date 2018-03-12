import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys, trimesh

def display_vox(fname):
    with open(fname,'r') as f:
        '''
        first 5 values are not voxel indices:
        refer to https://github.com/topskychen/voxelizer
        + 1 because voxelization done at dimension - 2; to get padding of 1 all around
        '''
        vox = np.array(f.read().replace('\n',' ').split(' ')[5:-1]).reshape((-1,3)).astype(int) + 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(vox[:,0],vox[:,1],vox[:,2],c='blue')
    plt.show()

def display_grid(DIM,fname):
    DIMmapping = {128:0,
                64:1,
                32:2,
                16:3,
                8:4,
                4:5}
    inputFull = np.load(fname, encoding='latin1')
    vox = np.where(inputFull[DIMmapping[DIM]] != 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(vox[0],vox[1],vox[2],c='blue')
    plt.show()

if __name__ == '__main__':
    fname_base = 'ModelNet10/' + sys.argv[1] + '/' + sys.argv[2] + '/' + sys.argv[1] + '_' + sys.argv[3]
    display_vox(fname_base + '.vox')
    DIM = 128
    if len(sys.argv)>4:
        DIM = int(sys.argv[4])
    display_grid(DIM,fname_base + '.npy')
    mesh = trimesh.load_mesh(fname_base + '.stl')
    mesh.show()
