import meshio
import os

def load_mesh(fname):
    dir_path=os.path.dirname(os.path.realpath(__file__))
    return meshio.read(os.join(dir_path,"meshes",fname))
