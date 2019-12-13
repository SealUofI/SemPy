import meshio
import os
import numpy as np

from functools import cmp_to_key

from sempy import debug
from sempy.quadrature import gauss_lobatto
from sempy.interpolation import lagrange
from sempy.kron import kron,kron_2d

class Face:
    def __init__(self,elem_id,face_id,nverts):
        self.elem_id=elem_id
        self.face_id=face_id
        self.nverts=nverts;
        self.neighbor_elem_id=-1;
        self.neighbor_face_id=-1
        self.verts=[]

    def get_nverts(self):
        return self.nverts

    def add_vert(self,v):
        self.verts.append(v)

    def get_verts(self):
        return self.verts

def compare_faces(f,g):
    if f.elem_id < g.elem_id: return -1
    if f.elem_id > g.elem_id: return  1
    if f.face_id < g.face_id: return -1
    if f.face_id > g.face_id: return  1
    return 0

def compare_verts(f,g):
    nverts=f.get_nverts()
    f_verts=f.get_verts()
    g_verts=g.get_verts()

    for i in range(nverts):
        if f_verts[i] < g_verts[i]: return -1
        if f_verts[i] > g_verts[i]: return  1
    return 0

class Mesh:
    def __init__(self,fname):
        if not isinstance(fname,str):
            raise Exception("Only strings are allowed for file name")
        self.read(fname)
        self.find_connectivities()

    def read(self,fname):
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

        ## setup face data
        self.nfaces     =2*self.ndim
        self.nface_verts=self.ndim+1
        if self.ndim==3:
            self.face_to_vert_map=np.array([[0,1,2,3],[0,1,5,4],\
                [1,2,6,5],[2,3,7,6],[3,0,4,7],[4,5,6,7]])
        else: #2D mesh
            raise Exception("2D not supported yet.")

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

        self.x=np.array(self.x).reshape((self.get_num_elements(),\
            self.get_num_verts()))
        self.y=np.array(self.y).reshape((self.get_num_elements(),\
            self.get_num_verts()))
        self.z=np.array(self.z).reshape((self.get_num_elements(),\
            self.get_num_verts()))

        ## TODO: Read in boundary faces
        if self.ndim==3:
            self.boundary_faces=meshin.cells['quad']
        else:
            self.boundary_faces=meshin.cells['line']

        self.num_boundary_faces=len(self.boundary_faces)

    def get_num_elements(self):
        return self.num_elements

    def get_ndim(self):
        return self.ndim

    def get_num_faces(self):
        return self.nfaces

    def get_num_verts(self):
        return self.num_vertices

    def get_face_to_vert_map(self):
        return self.face_to_vert_map

    def get_elem_to_vert_map(self):
        return self.elem_to_vert_map

    def get_num_face_verts(self):
        return self.nface_verts

    def find_connectivities(self):
        nelems     =self.get_num_elements()
        nfaces     =self.get_num_faces()
        nface_verts=self.get_num_face_verts()
        if debug:
            print("elems/faces/face_verts: {}/{}/{}".format(\
                nelems,nfaces,nface_verts))

        face_to_vert_map=self.get_face_to_vert_map()
        elem_to_vert_map=self.get_elem_to_vert_map()

        faces=[]
        for e in range(nelems):
            for f in range(nfaces):
                face=Face(e,f,nface_verts)
                for n in range(nface_verts):
                    vid=elem_to_vert_map[e,face_to_vert_map[f,n]]
                    face.add_vert(vid)
                face.get_verts().sort(reverse=True)
                faces.append(face)

        faces.sort(key=cmp_to_key(compare_verts))

        for i in range(nelems*nfaces-1):
            if not compare_verts(faces[i],faces[i+1]):
                if debug:
                    print("faces {}/{} and {}/{} match.".format(\
                        faces[i  ].elem_id,faces[i  ].face_id,
                        faces[i+1].elem_id,faces[i+1].face_id))
                faces[i  ].neighbor_elem_id=faces[i+1].elem_id
                faces[i  ].neighbor_face_id=faces[i+1].face_id
                faces[i+1].neighbor_elem_id=faces[i  ].elem_id
                faces[i+1].neighbor_face_id=faces[i  ].face_id
        faces=sorted(faces,key=cmp_to_key(compare_faces))

        self.elem_to_elem_map=[]
        self.elem_to_face_map=[]
        n=0
        for e in range(nelems):
            for f in range(nfaces):
                self.elem_to_elem_map.append(
                    faces[n].neighbor_elem_id)
                self.elem_to_face_map.append(
                    faces[n].neighbor_face_id)
                n=n+1
        self.elem_to_elem_map=\
            np.array(self.elem_to_elem_map).reshape((nelems,nfaces))
        self.elem_to_face_map=\
            np.array(self.elem_to_face_map).reshape((nelems,nfaces))

    def find_physical_nodes(self,N):
        self.N  =N;
        self.Nq =N+1;

        if self.get_ndim()==3:
            self.Nfp=(N+1)*(N+1);
            self.Np =(N+1)*(N+1)*(N+1)
        else:
            self.Nfp=(N+1)
            self.Np =(N+1)*(N+1)

        z_1,jnk=gauss_lobatto(1)
        z_N,jnk=gauss_lobatto(N)
        J  =lagrange(z_N,z_1)

        self.xe=np.zeros((self.get_num_elements(),self.Np))
        self.ye=np.zeros((self.get_num_elements(),self.Np))
        if self.get_ndim()==3:
            self.ze=np.zeros((self.get_num_elements(),self.Np))

        if self.ndim==3:
            for e in range(self.get_num_elements()):
                x=self.x[e,:]
                y=self.y[e,:]
                z=self.z[e,:]

                xx=np.array([x[0],x[1],x[3],x[2],x[4],x[5],x[7],x[6]])
                yy=np.array([y[0],y[1],y[3],y[2],y[4],y[5],y[7],y[6]])
                zz=np.array([z[0],z[1],z[3],z[2],z[4],z[5],z[7],z[6]])

                xe=kron(J,J,J,xx)
                ye=kron(J,J,J,yy)
                ze=kron(J,J,J,zz)

                self.xe[e,:]=xe
                self.ye[e,:]=ye
                self.ze[e,:]=ze
        else:
            for e in range(self.get_num_elements()):
                x=self.x[e,:]
                y=self.y[e,:]

                xx=np.array([x[0],x[1],x[3],x[2]])
                yy=np.array([y[0],y[1],y[3],y[2]])

                xe=kron(J,J,J,xx)
                ye=kron(J,J,J,yy)

                self.xe[e,:]=xe
                self.ye[e,:]=ye

    def get_x(self):
        return self.xe

    def get_y(self):
        return self.ye

    def get_z(self):
        return self.ze

def load_mesh(fname):
    dir_path=os.path.dirname(os.path.realpath(__file__))
    full_path=os.path.join(dir_path,"meshes",fname)
    return Mesh(full_path)
