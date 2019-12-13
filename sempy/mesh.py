import meshio
import os
import numpy as np

class Mesh:
    def __init__(self,fname):
        if not isinstance(fname,str):
            raise Exception("Only strings are allowed for file name")

        meshin=meshio.read(fname)

        ## element to vertex map
        self.elem_to_vert_map=meshin.cells['hexahedron']
        if len(self.elem_to_vert_map)==0: # 2D mesh
            self.elem_to_vert_map=meshin.cells['quad']
        assert len(self.elem_to_vert_map)>0

        ## number of elements
        self.num_elements=len(self.elem_to_vert_map[:,0])
        assert self.num_elements>0

        ## number of vertices
        self.num_vertices=len(self.elem_to_vert_map[0,:])

        ## dimension of the mesh
        assert len(meshin.points)>0
        self.ndim=len(meshin.points[0,:])

        ## set co-ordinates of the element vertices
        self.x=[]
        self.y=[]
        self.z=[]

        for i in range(self.num_elements):
            for j in range(self.num_vertices):
                self.x.append(meshin.points[\
                    self.elem_to_vert_map[i,j],0])
                self.y.append(meshin.points[\
                    self.elem_to_vert_map[i,j],1])
                if self.ndim==3:
                    self.z.append(meshin.points[\
                        self.elem_to_vert_map[i,j],2])

        self.x=np.array(self.x)
        self.y=np.array(self.y)
        self.z=np.array(self.z)

        ## Read in boundary faces
        if self.ndim==3:
            self.boundary_faces=meshin.cells['quad']
        else:
            self.boundary_faces=meshin.cells['line']

        self.num_boundary_faces=len(self.boundary_faces)

    def setup(self,N):
        self.N  =N;
        self.Nq =N+1;
        self.Nfp=(N+1)*(N+1);
        self.Np =(N+1)*(N+1)*(N+1)

    def get_num_elements(self):
        return self.num_elements

    def get_num_points(self):
        return self.num_elements*self.num_vertices

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

def load_mesh(fname):
    dir_path=os.path.dirname(os.path.realpath(__file__))
    full_path=os.path.join(dir_path,"meshes",fname)
    return Mesh(full_path)
