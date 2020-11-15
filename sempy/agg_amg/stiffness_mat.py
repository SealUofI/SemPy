import numpy as np
import scipy.sparse as sp
np.set_printoptions(threshold=np.inf)

def get_maskl(t):
    E = t.shape[0]
    ng = np.max(np.max(t)) + 1 # Ce n'est pas correct probablemente.

    etmp = t.T; etmp = np.vstack((etmp, etmp[0]))
    edge=np.zeros((6,E),dtype=np.int32)
    edge[0:2,:] = etmp[0:2,:];
    edge[2:4,:] = etmp[1:3,:];
    edge[4:6,:] = etmp[2:4,:];
    edge=np.reshape(edge,(3*E,2)).T;
    

    #edge = edge[np.lexsort(edge[0,:])];
    #edge=edge.T
    edge=np.sort(edge,axis=0); edge=edge.T;
    
    # Is this the same as matlab
    ind = np.lexsort((edge[:,1],edge[:,0]))
    print(ind)
    edge = edge[ind]
    #print(edge)
    #print(ind)
    #ind = np.argsort(edge, axis=0)
    #edge = edge[ind[:,0]]
    #np.take_along_axis(edge, ind, axis=0)
    #print(edge)
    #[edge,ind]=sortrows(edge); # FIXME
    #exit()

    nedge=edge.shape[0]; flag=np.zeros((nedge,));

    ## Mark non-isolated edges with flag=1
    for k in range(0,nedge-1):
        if np.all(edge[k,:]==edge[k+1,:]):
            flag[k]=flag[k]+1; 
            flag[k+1]=flag[k+1]+1;
    
    flag[ind]=flag;               
    ## Map edge flags back to original 3xE ordering.
    flag=np.reshape(flag,(3,E));

    ptr = np.zeros((2,3),dtype=np.int32);         ## Zero out local mask entries for any isolated edge
    ptr[0,0]=0; ptr[1,0]=1;
    ptr[0,1]=1; ptr[1,1]=2;
    ptr[0,2]=2; ptr[1,2]=0;

    maskL=np.ones((E,3));
    for e in range(E):
        for j in range(3):
            if flag[j,e]==0: 
                #print((j,e))
                maskL[e,ptr[:,j]]=0

    #
    #   Share boundary vertices flagged by isolated edges with neighboring elements
    #


    flag=np.ones((ng,));
    for e in range(E):
        for j in range(3):
            g = t[e,j]
            flag[g]=flag[g]*maskL[e,j];

    gbdry=np.zeros((ng,), dtype=np.int32);  nbdry=-1;
    for e in range(E):
        for j in range(3):
            g = t[e,j]
            maskL[e,j]=flag[g]
            if flag[g]==0:
                nbdry = nbdry+1
                gbdry[nbdry]=g

    gbdry = gbdry[:nbdry]
    gbdry = np.unique(gbdry);

    return maskL, gbdry

def fem_mat(p,t):
    nt = t.shape[0]
    nl = 3*nt

    y23 = p[t[:,1],1] - p[t[:,2],1]
    y31 = p[t[:,2],1] - p[t[:,0],1]
    y12 = p[t[:,0],1] - p[t[:,1],1]
    x32 = p[t[:,2],0] - p[t[:,1],0]
    x13 = p[t[:,0],0] - p[t[:,2],0]
    x21 = p[t[:,1],0] - p[t[:,0],0]
    area = 0.5*(x21*y31 - y12*x13)
    aream = np.min(area)
    areaM = np.max(area)

    eflip = np.flatnonzero(area<0)
    nflip = len(eflip)
    if nflip > 0:
        temp = t[eflip, 2]
        t[eflip,2] = t[eflip,3]
        t[eflip,3] = temp
    
    y23 = p[t[:,1],1] - p[t[:,2],1]
    y31 = p[t[:,2],1] - p[t[:,0],1]
    y12 = p[t[:,0],1] - p[t[:,1],1]
    x32 = p[t[:,2],0] - p[t[:,1],0]
    x13 = p[t[:,0],0] - p[t[:,2],0]
    x21 = p[t[:,1],0] - p[t[:,0],0]
    area4i = 1./area; area4i = 0.25*area4i;

    # Include endpoints or not?
    i0 = np.arange(0,nt)
    i1 = np.arange(1,nt+1)

    A1 = np.zeros((3,3,nt));  B1 = A1.copy();

    A1[0,0,:] = area4i*( y23*y23+x32*x32 );
    A1[0,1,:] = area4i*( y23*y31+x32*x13 );
    A1[0,2,:] = area4i*( y23*y12+x32*x21 );
    A1[1,0,:] = area4i*( y31*y23+x13*x32 );
    A1[1,1,:] = area4i*( y31*y31+x13*x13 );
    A1[1,2,:] = area4i*( y31*y12+x13*x21 );
    A1[2,0,:] = area4i*( y12*y23+x21*x32 );
    A1[2,1,:] = area4i*( y12*y31+x21*x13 );
    A1[2,2,:] = area4i*( y12*y12+x21*x21 );

    dmass=1         # Diagonal mass matrix
    dmass=0         # Full (local) mass matix
    if dmass==0: 
        B1[0,0,:] = area/6;
        B1[0,1,:] = area/12;
        B1[0,2,:] = area/12;
        B1[1,0,:] = area/12;
        B1[1,1,:] = area/6;
        B1[1,2,:] = area/12;
        B1[2,0,:] = area/12;
        B1[2,1,:] = area/12;
        B1[2,2,:] = area/6;
    else:
        B1[0,0,:] = area/3;
        B1[1,1,:] = area/3;
        B1[2,2,:] = area/3;

    # for e=0:nt-1;        THIS APPROACH IS WAY TOO SLOW
    #   AL(3*e+(1:3),3*e+(1:3)) = A1(:,:,e+1);
    #   BL(3*e+(1:3),3*e+(1:3)) = B1(:,:,e+1);
    # end;

    d0=np.zeros((3,nt));        # main diagonal
    d1=np.zeros((3,nt));        # 1st lower diagonal
    d2=np.zeros((3,nt));        # 2nd lower diagonal

    d0[0,:]=A1[0,0,:];
    d0[1,:]=A1[1,1,:];
    d0[2,:]=A1[2,2,:];     d0=np.reshape(d0,(nl,));
    d1[0,:]=A1[1,0,:];
    d1[1,:]=A1[2,1,:];     d1=np.reshape(d1,(nl,));
    d2[0,:]=A1[2,0,:];     d2=np.reshape(d2,(nl,));


    #AL = spalloc(nl,nl,9*nl); BL = AL.copy();
    AL=sp.diags((d2, d1),offsets=(-2,-1), shape=(nl,nl))
    AL=AL+AL.T
    Ad=sp.diags(d0,offsets=0,shape=(nl,nl))
    AL=AL+Ad;

    d0=np.zeros((3,nt));        # main diagonal
    d1=np.zeros((3,nt));        # 1st lower diagonal
    d2=np.zeros((3,nt));        # 2nd lower diagonal

    d0[0,:]=B1[0,0,:];
    d0[1,:]=B1[1,1,:];
    d0[2,:]=B1[2,2,:];     d0=np.reshape(d0,(nl,));
    d1[0,:]=B1[1,0,:];
    d1[1,:]=B1[2,1,:];     d1=np.reshape(d1,(nl,));
    d2[0,:]=B1[2,0,:];     d2=np.reshape(d2,(nl,));

    BL=sp.diags([d2, d1],offsets=(-2,-1),shape=(nl,nl));
    BL=BL+BL.T;
    Ad=sp.diags(d0,offsets=0,shape=(nl,nl));
    BL=BL+Ad

    Q  = sp.csr_matrix((np.ones((nl,)), (np.arange(nl),np.reshape(t.T, (nl,)))));

    return (AL, BL, Q)

def stiffness_mat(p,t):

    #  load c.xy;    p=c; load c.pts;   t=c;
    #  load a.xy;    p=a; load a.pts;   t=a;
    #  load pts.dat;    p=pts; load tri.dat;  t=tri;

    E, nv = t.shape
    t1 = np.unique(t.flatten())
    nL = E*nv
    nb = len(t1)
    xb = p[t1,0]
    yb=p[t1,1]

    AL, BL, Q = fem_mat(p,t)

    maskL,gbdry = get_maskl(t)
    for A in [maskL,gbdry]:
        print(A.shape)
        print(np.any(np.isnan(A)))
        print(np.linalg.norm(A))
        print()
    #exit()

    #   Here, add Dirichlet/Neumann discriminators, if desired.

    ### nbdry = size(gbdry,1);
    ### rbdry = sqrt(p(gbdry,1).^2+p(gbdry,2).^2);
    ### ibbox = find(rbdry > 0.41);
    ### %ibbox = find(rbdry < 0.41);
    ### dbdry = gbdry(ibbox);
    ### gbdry = dbdry;


    ### Set up restriction/permutation matrix to make boundary nodes last

    ir = np.arange(nb)
    n = nb-gbdry.shape[0]
    ir[gbdry] = ir[gbdry] + 2*nb

    ind = np.lexsort(ir); i_s = ir[ind]
    P = sp.eye(nb).tocsr(); P=P[:,ind]; R=P.T

    #n = nb-gbdry.shape[0]

    # Plotting code not implemented

    #for A in (AL, BL, Q, R):
    #    print(A.shape)
    #    #print(np.any(np.isnan(A)))
    #    print(np.linalg.norm(A))
    #    print()
       

    return (AL, BL, Q, R, xb, yb, p, t)

if __name__ ==  "__main__":
    pts = np.loadtxt("pts.dat")
    tri = np.loadtxt("tri.dat",dtype=np.int32) - 1
    
    stiffness_mat(pts,tri)
