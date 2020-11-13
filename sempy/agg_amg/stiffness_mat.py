import numpy as np


'''
def fem_mat(p,t):
    nt = t.shape[0]
    nl = 3*nt

    y23 = p[t[:,2],2] - p[t[:,3],2]
    y31 = p[t[:,3],2] - p[t[:,1],2]
    y12 = p[t[:,1],2] - p[t[:,2],2]
    x32 = p[t[:,3],1] - p[t[:,2],1]
    x13 = p[t[:,1],1] - p[t[:,3],1]
    x21 = p[t[:,2],1] - p[t[:,1],1]
    area = 0.5*(x21*y31 - y12*x13)
    aream = np.minimum(area)
    areaM = np.maximum(area)

    eflip = 
'''
'''
  eflip = find(area<0);      % Check for negative Jacobians
  nflip = length(eflip);
  if nflip > 0; 
     temp       = t(eflip,2);
     t(eflip,2) = t(eflip,3);
     t(eflip,3) = temp;
  end;

  y23 = p(t(:,2),2) - p(t(:,3),2);
  y31 = p(t(:,3),2) - p(t(:,1),2);
  y12 = p(t(:,1),2) - p(t(:,2),2);
  x32 = p(t(:,3),1) - p(t(:,2),1);
  x13 = p(t(:,1),1) - p(t(:,3),1);
  x21 = p(t(:,2),1) - p(t(:,1),1);
  area4i = 1./area; area4i = 0.25*area4i;



  i0 = (0:nt-1)';
  i1 = (1:nt)';
  AL = spalloc(nl,nl,9*nl); BL = AL;
  A1 = zeros(3,3,nt);       B1 = A1;

  A1(1,1,:) = area4i.*( y23.*y23+x32.*x32 );
  A1(1,2,:) = area4i.*( y23.*y31+x32.*x13 );
  A1(1,3,:) = area4i.*( y23.*y12+x32.*x21 );
  A1(2,1,:) = area4i.*( y31.*y23+x13.*x32 );
  A1(2,2,:) = area4i.*( y31.*y31+x13.*x13 );
  A1(2,3,:) = area4i.*( y31.*y12+x13.*x21 );
  A1(3,1,:) = area4i.*( y12.*y23+x21.*x32 );
  A1(3,2,:) = area4i.*( y12.*y31+x21.*x13 );
  A1(3,3,:) = area4i.*( y12.*y12+x21.*x21 );

  dmass=1;         % Diagonal mass matrix
  dmass=0;         % Full (local) mass matix
  if dmass==0; 
     B1(1,1,:) = area/6;
     B1(1,2,:) = area/12;
     B1(1,3,:) = area/12;
     B1(2,1,:) = area/12;
     B1(2,2,:) = area/6;
     B1(2,3,:) = area/12;
     B1(3,1,:) = area/12;
     B1(3,2,:) = area/12;
     B1(3,3,:) = area/6;
  else
     B1(1,1,:) = area/3;
     B1(2,2,:) = area/3;
     B1(3,3,:) = area/3;
  end;

% for e=0:nt-1;        THIS APPROACH IS WAY TOO SLOW
%   AL(3*e+(1:3),3*e+(1:3)) = A1(:,:,e+1);
%   BL(3*e+(1:3),3*e+(1:3)) = B1(:,:,e+1);
% end;

d0=zeros(3,nt);        % main diagonal
d1=zeros(3,nt);        % 1st lower diagonal
d2=zeros(3,nt);        % 2nd lower diagonal

d0(1,:)=A1(1,1,:);
d0(2,:)=A1(2,2,:);
d0(3,:)=A1(3,3,:);     d0=reshape(d0,nl,1);
d1(1,:)=A1(2,1,:);
d1(2,:)=A1(3,2,:);     d1=reshape(d1,nl,1);
d2(1,:)=A1(3,1,:);     d2=reshape(d2,nl,1);

AL=spdiags([d2 d1],-2:-1,nl,nl);
AL=AL+AL';
Ad=spdiags(d0,0:0,nl,nl);
AL=AL+Ad;

d0=zeros(3,nt);        % main diagonal
d1=zeros(3,nt);        % 1st lower diagonal
d2=zeros(3,nt);        % 2nd lower diagonal

d0(1,:)=B1(1,1,:);
d0(2,:)=B1(2,2,:);
d0(3,:)=B1(3,3,:);     d0=reshape(d0,nl,1);
d1(1,:)=B1(2,1,:);
d1(2,:)=B1(3,2,:);     d1=reshape(d1,nl,1);
d2(1,:)=B1(3,1,:);     d2=reshape(d2,nl,1);

BL=spdiags([d2 d1],-2:-1,nl,nl);
BL=BL+BL';
Ad=spdiags(d0,0:0,nl,nl);
BL=BL+Ad;


Q  = sparse(1:nl,reshape(t',nl,1),1);
''' 

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

    #AL, BL, Q = local_fem_mats(p,t) # Need to write this function
    #maskL,gbdry = get_maskl(t) # Need to write this function

    #   Here, add Dirichlet/Neumann discriminators, if desired.

    ### nbdry = size(gbdry,1);
    ### rbdry = sqrt(p(gbdry,1).^2+p(gbdry,2).^2);
    ### ibbox = find(rbdry > 0.41);
    ### %ibbox = find(rbdry < 0.41);
    ### dbdry = gbdry(ibbox);
    ### gbdry = dbdry;


    ### Set up restriction/permutation matrix to make boundary nodes last

    #ir = np.arange(

    #n = nb-gbdry.shape[0]

    # Plotting code not implemented

    #return (AL, BL, Q, R, xb, yb, p, t)

if __name__ ==  "__main__":
    pts = np.loadtxt("pts.dat")
    tri = np.loadtxt("tri.dat",dtype=np.int32) - 1
    
    stiffness_mat(pts,tri)
