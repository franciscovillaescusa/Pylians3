################################################################################
################################################################################

#This library contains the routines needed to compute correlation functions.

############## AVAILABLE ROUTINES ##############
####### Landy-Szalay estimator ########
#TPCF
      #DDR_pairs
           #distances_core
           #DR_distances
           #indexes_subbox_neigh
           #indexes_subbox
      #write_results
      #read_results
      #xi
#TPCCF
#create_random_catalogue
      #DD_file

##### Ariel Sanchez estimator ######
#all_distances_grid
#distances_grid

##### Taruya estimator #####


################################################

######## COMPILATION ##########
#If the library needs to be compiled type: 
#mpirun -np 2 python correlation_function_library.py compile
###############################

#IMPORTANT!! If the c/c++ functions need to be modified, the code has to be
#compiled by calling those functions within this file, otherwise it gives errors

################################################################################
################################################################################


from mpi4py import MPI
import numpy as np
import scipy.weave as wv
import sys,os
import time

###### MPI DEFINITIONS ######
comm=MPI.COMM_WORLD
nprocs=comm.Get_size()
myrank=comm.Get_rank()





################################################################################
#This functions computes the TPCF (2pt correlation function) 
#from an N-body simulation. It takes into account boundary conditions
#VARIABLES:
#pos_g: array containing the positions of the galaxies
#pos_r: array containing the positions of the random particles catalogue
#BoxSize: Size of the Box. Units must be equal to those of pos_r/pos_g
#DD_action: compute number of galaxy pairs from data or read them---compute/read
#RR_action: compute number of random pairs from data or read them---compute/read
#DR_action: compute number of galaxy-random pairs or read them---compute/read
#DD_name: file name to write/read galaxy-galaxy pairs results
#RR_name: file name to write/read random-random pairs results
#DR_name: file name to write/read galaxy-random pairs results
#bins: number of bins to compute the 2pt correlation function
#Rmin: minimum radius to compute the 2pt correlation function
#Rmax: maximum radius to compute the 2pt correlation function
#USAGE: at the end of the file there is a example of how to use this function
def TPCF(pos_g,pos_r,BoxSize,DD_action,RR_action,DR_action,
         DD_name,RR_name,DR_name,bins,Rmin,Rmax,verbose=False):

    #dims determined requiring that no more 8 adyacent subboxes will be taken
    dims=int(BoxSize/Rmax)
    dims2=dims**2; dims3=dims**3

    ##### MASTER #####
    if myrank==0:

        #compute the indexes of the halo/subhalo/galaxy catalogue
        Ng=len(pos_g)*1.0; indexes_g=[]
        coord=np.floor(dims*pos_g/BoxSize).astype(np.int32)
        index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]
        for i in range(dims3):
            ids=np.where(index==i)[0]
            indexes_g.append(ids)
        indexes_g=np.array(indexes_g)

        #compute the indexes of the random catalogue
        Nr=len(pos_r)*1.0; indexes_r=[]
        coord=np.floor(dims*pos_r/BoxSize).astype(np.int32)
        index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]
        for i in range(dims3):
            ids=np.where(index==i)[0]
            indexes_r.append(ids)
        indexes_r=np.array(indexes_r)


        #compute galaxy-galaxy pairs: DD
        if DD_action=='compute':
            DD=DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1=indexes_g,
                             indexes2=None,pos1=pos_g,pos2=None)
            if verbose:
                print(DD);  print(np.sum(DD))
            #write results to a file
            write_results(DD_name,DD,bins,'radial')
        else:
            #read results from a file
            DD,bins_aux=read_results(DD_name,'radial')
            if bins_aux!=bins:
                print('Sizes are different!');  sys.exit()

        #compute random-random pairs: RR
        if RR_action=='compute':
            RR=DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1=indexes_r,
                             indexes2=None,pos1=pos_r,pos2=None)
            if verbose:
                print(RR);  print(np.sum(RR))
            #write results to a file
            write_results(RR_name,RR,bins,'radial')
        else:
            #read results from a file
            RR,bins_aux=read_results(RR_name,'radial')
            if bins_aux!=bins:
                print('Sizes are different!');  sys.exit()

        #compute galaxy-random pairs: DR
        if DR_action=='compute':
            DR=DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1=indexes_g,
                         indexes2=indexes_r,pos1=pos_g,pos2=pos_r)
            if verbose:
                print(DR);  print(np.sum(DR))
            #write results to a file
            write_results(DR_name,DR,bins,'radial')
        else:
            #read results from a file
            DR,bins_aux=read_results(DR_name,'radial')
            if bins_aux!=bins:
                print('Sizes are different!');  sys.exit()


        #final procesing
        bins_histo=np.logspace(np.log10(Rmin),np.log10(Rmax),bins+1)
        middle=0.5*(bins_histo[:-1]+bins_histo[1:])
        DD*=1.0; RR*=1.0; DR*=1.0

        r,xi_r,error_xi_r=[],[],[]
        for i in range(bins):
            if (RR[i]>0.0): #avoid divisions by 0
                xi_aux,error_xi_aux=xi(DD[i],RR[i],DR[i],Ng,Nr)
                r.append(middle[i])
                xi_r.append(xi_aux)
                error_xi_r.append(error_xi_aux)

        r=np.array(r);  xi_r=np.array(xi_r);  error_xi_r=np.array(error_xi_r)
        return r,xi_r,error_xi_r



    ##### SLAVES #####
    else:
        if DD_action=='compute':
            DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,
                      indexes1=None,indexes2=None,pos1=None,pos2=None)
        if RR_action=='compute':
            DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,
                      indexes1=None,indexes2=None,pos1=None,pos2=None)          
        if DR_action=='compute':
            DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,
                      indexes1=None,indexes2=None,pos1=None,pos2=None)

        return None,None,None


########################################################################     
############### COMPUTE THE NUMBER OF PAIRS IN A CATALOG ###############
########################################################################
def DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1,indexes2,pos1,pos2):

    dims2=dims**2; dims3=dims**3

    #we put bins+1. The last bin is only for pairs separated by r=Rmax
    pairs=np.zeros(bins+1,dtype=np.int64) 

    ##### MASTER #####
    if myrank==0:
        #Master sends the indexes and particle positions to the slaves
        for i in range(1,nprocs):
            comm.send(pos1,dest=i,tag=6)
            comm.send(pos2,dest=i,tag=7)
            comm.send(indexes1,dest=i,tag=8)
            comm.send(indexes2,dest=i,tag=9)

        #Masters distributes the calculation among slaves
        for subbox in range(dims3):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(False,dest=b,tag=2)
            comm.send(subbox,dest=b,tag=3)

        #Master gathers partial results from slaves and returns the final result
        for j in range(1,nprocs):
            b=comm.recv(source=MPI.ANY_SOURCE,tag=1)
            comm.send(True,dest=b,tag=2)
            pairs_aux=comm.recv(source=b,tag=10)
            pairs+=pairs_aux

        #the last element is just for situations in which r=Rmax
        pairs[bins-1]+=pairs[bins]

        return pairs[:-1]


    ##### SLAVES #####
    else:
        #position of the center of each subbox
        sub_c=np.empty(3,dtype=np.float32)

        #slaves receive the positions and indexes
        pos1=comm.recv(source=0,tag=6)
        pos2=comm.recv(source=0,tag=7)
        indexes1=comm.recv(source=0,tag=8)
        indexes2=comm.recv(source=0,tag=9)

        comm.send(myrank,dest=0,tag=1)
        final=comm.recv(source=0,tag=2)
        while not(final):
            
            subbox=comm.recv(source=0,tag=3)
            core_ids=indexes1[subbox] #ids of the particles in the subbox
            pos0=pos1[core_ids]

            sub_c[0]=(subbox//dims2+0.5)*BoxSize/dims
            sub_c[1]=((subbox%dims2)//dims+0.5)*BoxSize/dims
            sub_c[2]=((subbox%dims2)%dims+0.5)*BoxSize/dims

            #galaxy-galaxy or random-random case
            if pos2==None: 
                #first: distances between particles in the same subbox
                distances_core(pos0,BoxSize,bins,Rmin,Rmax,pairs)

                #second: distances between particles in the subbox and particles around
                ids=indexes_subbox_neigh(sub_c,Rmax,dims,BoxSize,indexes1,subbox)
                if ids!=[]:
                    posN=pos1[ids]
                    DR_distances(pos0,posN,BoxSize,bins,Rmin,Rmax,pairs)

            #galaxy-random case
            else:          
                ids=indexes_subbox(sub_c,Rmax,dims,BoxSize,indexes2)
                posN=pos2[ids]
                DR_distances(pos0,posN,BoxSize,bins,Rmin,Rmax,pairs)

            comm.send(myrank,dest=0,tag=1)
            final=comm.recv(source=0,tag=2)

        #print 'cpu ',myrank,' finished: transfering data to master'
        comm.send(pairs,dest=0,tag=10)
########################################################################
#This function writes partial results to a file
def write_results(fname,histogram,bins,case):
    f=open(fname,'w')
    if case=='par-perp':
        for i in range(len(histogram)):
            coord_perp=i//bins
            coord_par=i%bins
            f.write(str(coord_par)+' '+str(coord_perp)+' '+str(histogram[i])+'\n')
    elif case=='radial':
        for i in range(len(histogram)):
            f.write(str(i)+' '+str(histogram[i])+'\n')
    else:
        print('Error in the description of case:')
        print('Choose between: par-perp or radial')
    f.close()        
########################################################################
#This functions reads partial results of a file
def read_results(fname,case):

    histogram=[]

    if case=='par-perp':
        bins=np.around(np.sqrt(size)).astype(np.int64)

        if bins*bins!=size:
            print('Error finding the size of the matrix'); sys.exit()

        f=open(fname,'r')
        for line in f.readlines():
            a=line.split()
            histogram.append(int(a[2]))
        f.close()
        histogram=np.array(histogram)
        return histogram,bins
    elif case=='radial':
        f=open(fname,'r')
        for line in f.readlines():
            a=line.split()
            histogram.append(int(a[1]))
        f.close()
        histogram=np.array(histogram)
        return histogram,histogram.shape[0]
    else:
        print('Error in the description of case:')
        print('Choose between: par-perp or radial')
########################################################################
#this function computes the distances between all the particles-pairs and
#return the number of pairs found in each distance bin
def distances_core(pos,BoxSize,bins,Rmin,Rmax,pairs):

    l=pos.shape[0]

    support = """
       #include <iostream>
       using namespace std;
    """
    code = """
       float middle=BoxSize/2.0;
       float dx,dy,dz,r;
       float x1,y1,z1,x2,y2,z2;
       float delta=log10(Rmax/Rmin)/bins;
       int bin,i,j;

       for (i=0;i<l;i++){
            x1=pos(i,0);
            y1=pos(i,1);
            z1=pos(i,2);
            for (j=i+1;j<l;j++){
                x2=pos(j,0);
                y2=pos(j,1);
                z2=pos(j,2);
                dx=(fabs(x1-x2)<middle) ? x1-x2 : BoxSize-fabs(x1-x2);
                dy=(fabs(y1-y2)<middle) ? y1-y2 : BoxSize-fabs(y1-y2);
                dz=(fabs(z1-z2)<middle) ? z1-z2 : BoxSize-fabs(z1-z2);
                r=sqrt(dx*dx+dy*dy+dz*dz);

               if (r>=Rmin && r<=Rmax){
                   bin=(int)(log10(r/Rmin)/delta);
                   pairs(bin)+=1; 
               }
            }   
       }
    """
    wv.inline(code,['pos','l','BoxSize','Rmin','Rmax','bins','pairs'],
              type_converters = wv.converters.blitz,
              support_code = support,libraries = ['m'])

    return pairs
########################################################################
#pos1---an array of positions
#pos2---an array of positions
#the function returns the number of pairs in distance bins between pos1 and pos2
def DR_distances(p1,p2,BoxSize,bins,Rmin,Rmax,pairs):

    l1=p1.shape[0]
    l2=p2.shape[0]

    support = """
       #include <iostream>
       using namespace std;
    """
    code = """
       float middle=BoxSize/2.0;
       float dx,dy,dz,r;
       float x1,y1,z1,x2,y2,z2;
       float delta=log10(Rmax/Rmin)/bins;
       int bin,i,j;

       for (i=0;i<l1;i++){
           x1=p1(i,0); 
           y1=p1(i,1);
           z1=p1(i,2);
           for (j=0;j<l2;j++){
               x2=p2(j,0); 
               y2=p2(j,1);
               z2=p2(j,2);
               dx=(fabs(x1-x2)<middle) ? x1-x2 : BoxSize-fabs(x1-x2);
               dy=(fabs(y1-y2)<middle) ? y1-y2 : BoxSize-fabs(y1-y2);
               dz=(fabs(z1-z2)<middle) ? z1-z2 : BoxSize-fabs(z1-z2);
               r=sqrt(dx*dx+dy*dy+dz*dz);

               if (r>=Rmin && r<=Rmax){
                   bin=(int)(log10(r/Rmin)/delta);
                   pairs(bin)+=1; 
               }
           }   
       }
    """
    wv.inline(code,['p1','p2','l1','l2','BoxSize','Rmin','Rmax','bins','pairs'],
              type_converters = wv.converters.blitz,
              support_code = support)

    return pairs
########################################################################
#this routine computes the IDs of all the particles within the neighboord cells
#that which can lie within the radius Rmax
def indexes_subbox(pos,Rmax,dims,BoxSize,indexes):

    #we add dims to avoid negative numbers. For example
    #if something hold between -1 and 5, the array to be
    #constructed should have indexes -1 0 1 2 3 4 5. 
    #To achieve this in a clever way we add dims
    i_min=int(np.floor((pos[0]-Rmax)*dims/BoxSize))+dims
    i_max=int(np.floor((pos[0]+Rmax)*dims/BoxSize))+dims
    j_min=int(np.floor((pos[1]-Rmax)*dims/BoxSize))+dims
    j_max=int(np.floor((pos[1]+Rmax)*dims/BoxSize))+dims
    k_min=int(np.floor((pos[2]-Rmax)*dims/BoxSize))+dims
    k_max=int(np.floor((pos[2]+Rmax)*dims/BoxSize))+dims
    
    i_array=np.arange(i_min,i_max+1)%dims
    j_array=np.arange(j_min,j_max+1)%dims
    k_array=np.arange(k_min,k_max+1)%dims

    PAR_indexes=np.array([])
    for i in i_array:
        for j in j_array:
            for k in k_array:
                num=dims**2*i+dims*j+k
                ids=indexes[num]
                PAR_indexes=np.concatenate((PAR_indexes,ids)).astype(np.int32)

    return PAR_indexes
########################################################################
#this routine returns the ids of the particles in the neighboord cells
#that havent been already selected
def indexes_subbox_neigh(pos,Rmax,dims,BoxSize,indexes,subbox):

    #we add dims to avoid negative numbers. For example
    #if something hold between -1 and 5, the array to be
    #constructed should have indexes -1 0 1 2 3 4 5. 
    #To achieve this in a clever way we add dims
    i_min=int(np.floor((pos[0]-Rmax)*dims/BoxSize))+dims
    i_max=int(np.floor((pos[0]+Rmax)*dims/BoxSize))+dims
    j_min=int(np.floor((pos[1]-Rmax)*dims/BoxSize))+dims
    j_max=int(np.floor((pos[1]+Rmax)*dims/BoxSize))+dims
    k_min=int(np.floor((pos[2]-Rmax)*dims/BoxSize))+dims
    k_max=int(np.floor((pos[2]+Rmax)*dims/BoxSize))+dims
    
    i_array=np.arange(i_min,i_max+1)%dims
    j_array=np.arange(j_min,j_max+1)%dims
    k_array=np.arange(k_min,k_max+1)%dims

    ids=np.array([])
    for i in i_array:
        for j in j_array:
            for k in k_array:
                num=dims**2*i+dims*j+k
                if num>subbox:
                    ids_subbox=indexes[num]
                    ids=np.concatenate((ids,ids_subbox)).astype(np.int32)
    return ids
########################################################################
#This function computes the correlation function and its error once the number
#of galaxy-galaxy, random-random & galaxy-random pairs are given together
#with the total number of galaxies and random points
def xi(GG,RR,GR,Ng,Nr):
    
    normGG=2.0/(Ng*(Ng-1.0))
    normRR=2.0/(Nr*(Nr-1.0))
    normGR=1.0/(Ng*Nr)

    GGn=GG*normGG
    RRn=RR*normRR
    GRn=GR*normGR
    
    xi=GGn/RRn-2.0*GRn/RRn+1.0

    fact=normRR/normGG*RR*(1.0+xi)+4.0/Ng*(normRR*RR/normGG*(1.0+xi))**2
    err=normGG/(normRR*RR)*np.sqrt(fact)
    err=err*np.sqrt(3.0)

    return xi,err
########################################################################
################################################################################


################################################################################
#This functions computes the TPCCF (2pt cross-correlation function) 
#from an N-body simulation. It takes into account boundary conditions
#VARIABLES:
#pos_g1: array containing the positions of the galaxies1
#pos_g2: array containing the positions of the galaxies2
#pos_r: array containing the positions of the random particles catalogue
#BoxSize: Size of the Box. Units must be equal to those of pos_r/pos_g1/pos_g2
#DD_action: compute number of galaxy pairs from data or read them---compute/read
#RR_action: compute number of random pairs from data or read them---compute/read
#DR_action: compute number of galaxy-random pairs or read them---compute/read
#DD_name: file name to write/read galaxy-galaxy pairs results
#RR_name: file name to write/read random-random pairs results
#DR_name: file name to write/read galaxy-random pairs results
#bins: number of bins to compute the 2pt correlation function
#Rmin: minimum radius to compute the 2pt correlation function
#Rmax: maximum radius to compute the 2pt correlation function
#USAGE: at the end of the file there is a example of how to use this function
def TPCCF(pos_g1,pos_g2,pos_r,BoxSize,
          D1D2_action,D1R_action,D2R_action,RR_action,
          D1D2_name,D1R_name,D2R_name,RR_name,
          bins,Rmin,Rmax,verbose=False):          
          

    #dims determined requiring that no more 8 adyacent subboxes will be taken
    dims=int(BoxSize/Rmax)
    dims2=dims**2; dims3=dims**3

    ##### MASTER #####
    if myrank==0:

        #compute the indexes of the halo1/subhalo1/galaxy1 catalogue
        Ng1=len(pos_g1)*1.0; indexes_g1=[]
        coord=np.floor(dims*pos_g1/BoxSize).astype(np.int32)
        index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]
        for i in range(dims3):
            ids=np.where(index==i)[0]
            indexes_g1.append(ids)
        indexes_g1=np.array(indexes_g1)

        #compute the indexes of the halo2/subhalo2/galaxy2 catalogue
        Ng2=len(pos_g2)*1.0; indexes_g2=[]
        coord=np.floor(dims*pos_g2/BoxSize).astype(np.int32)
        index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]
        for i in range(dims3):
            ids=np.where(index==i)[0]
            indexes_g2.append(ids)
        indexes_g2=np.array(indexes_g2)

        #compute the indexes of the random catalogue
        Nr=len(pos_r)*1.0; indexes_r=[]
        coord=np.floor(dims*pos_r/BoxSize).astype(np.int32)
        index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]
        for i in range(dims3):
            ids=np.where(index==i)[0]
            indexes_r.append(ids)
        indexes_r=np.array(indexes_r)


        #compute galaxy1-galaxy2 pairs: D1D2
        if D1D2_action=='compute':
            D1D2=DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1=indexes_g1,
                           indexes2=indexes_g2,pos1=pos_g1,pos2=pos_g2)
            if verbose:
                print(D1D2)
                print(np.sum(D1D2))
            #write results to a file
            write_results(D1D2_name,D1D2,bins,'radial')
        else:
            #read results from a file
            D1D2,bins_aux=read_results(D1D2_name,'radial')
            if bins_aux!=bins:
                print('Sizes are different!')
                sys.exit()

        #compute galaxy1-random pairs: D1R
        if D1R_action=='compute':
            D1R=DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1=indexes_g1,
                          indexes2=indexes_r,pos1=pos_g1,pos2=pos_r)
            if verbose:
                print(D1R)
                print(np.sum(D1R))
            #write results to a file
            write_results(D1R_name,D1R,bins,'radial')
        else:
            #read results from a file
            D1R,bins_aux=read_results(D1R_name,'radial')
            if bins_aux!=bins:
                print('Sizes are different!')
                sys.exit()

        #compute galaxy2-random pairs: D2R
        if D2R_action=='compute':
            D2R=DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1=indexes_g2,
                          indexes2=indexes_r,pos1=pos_g2,pos2=pos_r)
            if verbose:
                print(D2R)
                print(np.sum(D2R))
            #write results to a file
            write_results(D2R_name,D2R,bins,'radial')
        else:
            #read results from a file
            D2R,bins_aux=read_results(D2R_name,'radial')
            if bins_aux!=bins:
                print('Sizes are different!')
                sys.exit()

        #compute random-random pairs: RR
        if RR_action=='compute':
            RR=DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1=indexes_r,
                         indexes2=None,pos1=pos_r,pos2=None)
            if verbose:
                print(RR)
                print(np.sum(RR))
            #write results to a file
            write_results(RR_name,RR,bins,'radial')
        else:
            #read results from a file
            RR,bins_aux=read_results(RR_name,'radial')
            if bins_aux!=bins:
                print('Sizes are different!')
                sys.exit()


        #final procesing
        bins_histo=np.logspace(np.log10(Rmin),np.log10(Rmax),bins+1)
        middle=0.5*(bins_histo[:-1]+bins_histo[1:])

        inside=np.where(RR>0)[0]
        D1D2=D1D2[inside]; D1R=D1R[inside]; D2R=D2R[inside]; RR=RR[inside]
        middle=middle[inside]

        D1D2n=D1D2*1.0/(Ng1*Ng2)
        D1Rn=D1R*1.0/(Ng1*Nr)
        D2Rn=D2R*1.0/(Ng2*Nr)
        RRn=RR*2.0/(Nr*(Nr-1.0))
        
        xi_r=D1D2n/RRn-D1Rn/RRn-D2Rn/RRn+1.0

        return middle,xi_r



    ##### SLAVES #####
    else:
        if D1D2_action=='compute':
            DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,
                      indexes1=None,indexes2=None,pos1=None,pos2=None)
        if D1R_action=='compute':
            DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,
                      indexes1=None,indexes2=None,pos1=None,pos2=None)
        if D2R_action=='compute':
            DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,
                      indexes1=None,indexes2=None,pos1=None,pos2=None)
        if RR_action=='compute':
            DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,
                      indexes1=None,indexes2=None,pos1=None,pos2=None) 

        return None,None
################################################################################


################################################################################
#This function returns the positions of the points comprising the random 
#catalogue. The function will check whether a random catalogue with the 
#specified number of random particles already exists. If so it will just use 
#that catalogue. We notice that this random catalogue only contains the 
#positions of the particles in box of size 1, so can be used for any BoxSize.
#The function will also check whether there is a file containing
#the number of random pairs (that depend on BoxSize, Rmin, Rmax and bins). If
#so it will use that file, otherwise it will compute the number of random pairs
#and write the result to a file. The name of the above two files is set 
#internally.
#random_points --------------> number of random points in the random catalogue
#Rmin -----------------------> minimum distance to compute the CF
#Rmax -----------------------> maximum distance to compute the CF
#bins -----------------------> number of bins to compute the CF
#BoxSize --------------------> size of the box containing the random catalogue 
def create_random_catalogue(random_points,Rmin,Rmax,bins,BoxSize):

    #set names of the files containing the random points and the random pairs
    rp_s = str(random_points)
    random_catalogue_fname  = 'random_catalogue_'+rp_s+'.dat'
    RR_name                 = 'RR_'+rp_s+'_'+str(Rmin)+'_'+str(Rmax)+'_'+\
                               str(bins)+'_'+str(BoxSize)+'.dat'
                          
    #create/read the random catalogue depending on whether it already exists
    if myrank==0:
        if not(os.path.exists(random_catalogue_fname+'.npy')):
            print('\nCreating random catalogue...')
            pos_r = np.random.random((random_points,3)).astype(np.float32)
            np.save(random_catalogue_fname,pos_r)
            pos_r *= BoxSize  #multiply for the correct units
        else:
            print('\nReading random catalogue...')
            pos_r = np.load(random_catalogue_fname+'.npy')*BoxSize #Mpc/h
        print('done')

    #compute the random pairs if file containing it does not exists
    if not(os.path.exists(RR_name)):
        if myrank==0:    print('\nComputing number of random pairs...')
        else:            pos_r=None
        DD_file(pos_r,BoxSize,RR_name,bins,Rmin,Rmax)
        if myrank==0:    print('done!!')
    else:
        if myrank==0:
            print('\nI found the file '+RR_name)
            print('I assume this file contains the number of random pairs in '+\
                  'a box of '+str(BoxSize)+' using '+str(bins)+\
                  ' bins between '+str(Rmin)+' and '+str(Rmax)+' for a '+\
                  'catalogue containing '+str(random_points)+' points')

    if myrank==0:            
        return pos_r,RR_name
    else:
        return None,RR_name        
########################################################################
#This function is used to compute the DD file (the number of random pairs in a
#random catalogue) that it is need for massive computation of the 2pt 
#correlation function
#from an N-body simulation. It takes into account boundary conditions
#VARIABLES:
#pos_r: array containing the positions of the random particles catalogue
#BoxSize: Size of the Box. Units must be equal to those of pos_r/pos_g
#RR_name: file name to write/read random-random pairs results
#bins: number of bins to compute the 2pt correlation function
#Rmin: minimum radius to compute the 2pt correlation function
#Rmax: maximum radius to compute the 2pt correlation function
#USAGE: at the end of the file there is a example of how to use this function
def DD_file(pos_r,BoxSize,RR_name,bins,Rmin,Rmax):

    #dims determined requiring that no more 8 adyacent subboxes will be taken
    dims=int(BoxSize/Rmax)
    dims2=dims**2; dims3=dims**3

    ##### MASTER #####
    if myrank==0:

        #compute the indexes of the random catalogue
        Nr=len(pos_r)*1.0; indexes_r=[]
        coord=np.floor(dims*pos_r/BoxSize).astype(np.int32)
        index=dims2*coord[:,0]+dims*coord[:,1]+coord[:,2]
        for i in range(dims3):
            ids=np.where(index==i)[0]
            indexes_r.append(ids)
        indexes_r=np.array(indexes_r)

        #compute random-random pairs: RR
        RR=DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,indexes1=indexes_r,
                     indexes2=None,pos1=pos_r,pos2=None)
        print(RR)
        print(np.sum(RR))
        #write results to a file
        write_results(RR_name,RR,bins,'radial')

    ##### SLAVES #####
    else:
        DDR_pairs(bins,Rmin,Rmax,BoxSize,dims,
                  indexes1=None,indexes2=None,pos1=None,pos2=None)           
################################################################################




################################################################################
#this function computes all the distances between the particles and the 
#values of delta(i)*delta(j) for the pairs. It can be computed serially with one
#single core or via several cores through omp
#pos ---------------------> positions of the particles
#delta -------------------> array containing the delta = (n - <n>) / <n>
#BoxSize -----------------> Size of the simulation box
#bins --------------------> number of bins to compute the correlation function
#Rmin --------------------> minimum value to compute xi(r)
#Rmax --------------------> maximum value to compute xi(r)
#pairs -------------------> array containing the number of pairs
#xi ----------------------> array containing the sum delta(i)*delta(j)
#N1 ----------------------> number of the particle from which start
#N2 ----------------------> number of the particle to finish
#serial ------------------> True for serial (1 core) and False for several cores
#threads -----------------> Number of cpus to be used for omp
def all_distances_grid(pos,delta,BoxSize,bins,Rmin,Rmax,pairs,xi,N1,N2,
                       serial=True,threads=1):

    l=pos.shape[0]

    support_serial = """
       #include <iostream>
       using namespace std;
    """
    support_omp = """
       #include <iostream>
       #include<omp.h>
       using namespace std;
    """

    code_serial = """
       float middle=BoxSize/2.0; 
       float dx,dy,dz,r,x1,y1,z1,x2,y2,z2;
       float delta_r=log10(Rmax/Rmin)/bins;
       int bin,i,j;

       for (i=N1;i<N2;i++){
           x1=pos(i,0); y1=pos(i,1); z1=pos(i,2);
           for (j=i+1;j<l;j++){
               x2=pos(j,0); y2=pos(j,1); z2=pos(j,2);
               dx=(fabs(x1-x2)<middle) ? x1-x2 : BoxSize-fabs(x1-x2);
               dy=(fabs(y1-y2)<middle) ? y1-y2 : BoxSize-fabs(y1-y2);
               dz=(fabs(z1-z2)<middle) ? z1-z2 : BoxSize-fabs(z1-z2);
               r=sqrt(dx*dx+dy*dy+dz*dz);

               if (r>=Rmin && r<=Rmax){
                   bin=(int)(log10(r/Rmin)/delta_r);
                   pairs(bin)+=1; xi(bin)+=delta(i)*delta(j);
               }
           }   
       }
    """
    code_omp = """
       omp_set_num_threads(threads);
       float middle=BoxSize/2.0;
       float dx,dy,dz,r,x1,y1,z1,x2,y2,z2;
       float delta_r=log10(Rmax/Rmin)/bins;
       int bin,i,j;

       #pragma omp parallel for private(x1,y1,z1,x2,y2,z2,dx,dy,dz,r,bin,j,N1,N2) shared(pairs,xi) 
       for (i=N1;i<N2;i++){
           x1=pos(i,0); y1=pos(i,1); z1=pos(i,2);
           for (j=i+1;j<l;j++){
               x2=pos(j,0); y2=pos(j,1); z2=pos(j,2);
               dx=(fabs(x1-x2)<middle) ? x1-x2 : BoxSize-fabs(x1-x2);
               dy=(fabs(y1-y2)<middle) ? y1-y2 : BoxSize-fabs(y1-y2);
               dz=(fabs(z1-z2)<middle) ? z1-z2 : BoxSize-fabs(z1-z2);
               r=sqrt(dx*dx+dy*dy+dz*dz);

               if (r>=Rmin && r<=Rmax){
                   bin=(int)(log10(r/Rmin)/delta_r);
                   #pragma omp atomic
                       pairs(bin)+=1; 
                   #pragma omp atomic
                       xi(bin)+=delta(i)*delta(j);
               }
           }   
       }
    """

    if serial==True:
        wv.inline(code_serial,
                  ['pos','delta','l','BoxSize','Rmin','Rmax','bins','pairs',
                   'xi','N1','N2'],
                  type_converters = wv.converters.blitz,
                  support_code = support_serial,libraries = ['m'])
    else:
        wv.inline(code_omp,
                  ['pos','delta','l','BoxSize','Rmin','Rmax','bins','pairs',
                   'xi','N1','N2','threads'],
                  type_converters = wv.converters.blitz, 
                  extra_compile_args=['-O3 -fopenmp'],
                  extra_link_args=['-lgomp'],
                  support_code = support_omp,libraries = ['m','gomp'])

    return pairs
################################################################################



################################################################################
#This function computes the correlation function from the points in a grid
#from the particle N1 to the particle N2. The distances between the pairs have
#to be computed previously and the indexes_coord are used to iterate only among
#the particles contributing to the correlation function.
#N1 ----------------------> number of the particle from which start
#N2 ----------------------> number of the particle to finish
#dims --------------------> number of points in the grid per dimension
#delta -------------------> array containing the delta = (n - <n>) / <n>
#pairs -------------------> array containing the number of pairs 
#xi ----------------------> array containing the sum delta(i)*delta(j)
#indexes_coord -----------> array with the i/j/k indexes of the neighbors
#indexes_distances -------> array with the bins for distances between neighbors
def distances_grid(N1,N2,dims,delta,pairs,xi,indexes_coord,indexes_distances):

    #perform a sanity checks
    if len(indexes_coord)!=len(indexes_distances):
        print('length of coord and distances indexes are different!!!')
        sys.exit()

    #for every particle we have to iterate among l neighbors contributing to xi
    l=len(indexes_coord)

    support_serial = """
       #include <iostream>
       using namespace std;
    """
    code_serial = """
       int i,j,index,index_x,index_y,index_z,id_x,id_y,id_z,bin;
       int dims2=dims*dims;
       float cont_i;

       for (i=N1;i<N2;i++){
           index_x = (i/dims2)%dims;
           index_y = (i/dims)%dims;
           index_z = (i%dims);
           cont_i  = delta(i);

           for (j=0;j<l;j++){
               id_x  = (index_x + indexes_coord(j,0))%dims;
               id_y  = (index_y + indexes_coord(j,1))%dims;
               id_z  = (index_z + indexes_coord(j,2))%dims;
               index = dims2*id_x + dims*id_y + id_z;
               if (index>i){
                   bin = indexes_distances(j);
                   pairs(bin) += 1;
                   xi(bin)    += cont_i*delta(index);
               }
           }   
       }
    """

    wv.inline(code_serial,
              ['N1','N2','dims','delta','xi','indexes_coord',
               'indexes_distances','l','pairs'],
              type_converters = wv.converters.blitz,
              extra_compile_args=['-O3'],
              support_code = support_serial,libraries = ['m'])

    return pairs
################################################################################
















############################### EXAMPLE OF USAGE ##############################
if len(sys.argv)==2:
    if sys.argv[0]=='correlation_function_library.py' and \
            sys.argv[1]=='compile':

        ###############################################################    
        ### compute the CF defining a random catalog ###
        points_g = 15000 
        points_r = 20000

        BoxSize = 500.0  #Mpc/h
        Rmin    = 1.0    #Mpc/h
        Rmax    = 50.0   #Mpc/h
        bins    = 30
        
        DD_action = 'compute';  RR_action = 'compute';   DR_action = 'compute'
        DD_name   = 'DD.dat';     RR_name = 'RR.dat';    DR_name   = 'DR.dat'
        
        pos_r = None;  pos_g = None
        
        if myrank==0:
            pos_g = np.random.random((points_g,3))*BoxSize
            pos_r = np.random.random((points_r,3))*BoxSize
            start=time.time()

        r,xi_r,error_xi = TPCF(pos_g,pos_r,BoxSize,
                               DD_action,RR_action,DR_action,
                               DD_name,RR_name,DR_name,
                               bins,Rmin,Rmax,verbose=True)
        
        if myrank==0:
            print(r);  print(xi_r);  print(error_xi)
            end=time.time();   print('time:',end-start)

        ###############################################################    
        ### compute the CF without defining a random catalog ###
        points_g = 15000 
        points_r = 20000

        BoxSize = 500.0  #Mpc/h
        Rmin    = 1.0    #Mpc/h
        Rmax    = 50.0   #Mpc/h
        bins    = 30

        #get the random particles positions reading/creating a random catalogue
        pos_r,RR_name = create_random_catalogue(points_r,Rmin,Rmax,bins,BoxSize)
        
        DD_action = 'compute';   DR_action = 'compute'
        DD_name   = 'DD.dat';    DR_name   = 'DR.dat'
        RR_action = 'read'
    
        pos_g = None
        
        if myrank==0:
            pos_g = np.random.random((points_g,3))*BoxSize
            start=time.time()

        r,xi_r,error_xi = TPCF(pos_g,pos_r,BoxSize,
                               DD_action,RR_action,DR_action,
                               DD_name,RR_name,DR_name,
                               bins,Rmin,Rmax,verbose=True)
        
        if myrank==0:
            print(r);  print(xi_r);  print(error_xi)
            end=time.time();   print('time:',end-start)

        ###############################################################    
        ### compute the CCF defining a random catalog ###
        points_g1 = 15000
        points_g2 = 15000
        points_r  = 20000

        BoxSize = 500.0  #Mpc/h
        Rmin    = 1.0    #Mpc/h
        Rmax    = 50.0   #Mpc/h
        bins    = 30

        D1D2_action = 'compute';  D1D2_name = 'D1D2.dat'
        D1R_action  = 'compute';  D1R_name  = 'D1R.dat'
        D2R_action  = 'compute';  D2R_name  = 'D2R.dat'
        RR_action   = 'compute';  RR_name   = 'RR.dat'

        pos_g1 = None;  pos_g2 = None;  pos_r = None

        if myrank==0:
            pos_g1 = np.random.random((points_g1,3))*BoxSize
            pos_g2 = np.random.random((points_g2,3))*BoxSize
            pos_r  = np.random.random((points_r,3))*BoxSize

        r,xi_r = TPCCF(pos_g1,pos_g2,pos_r,BoxSize,
                       D1D2_action,D1R_action,D2R_action,RR_action,
                       D1D2_name,D1R_name,D2R_name,RR_name,
                       bins,Rmin,Rmax,verbose=True)

        if myrank == 0:
            print(r);  print(xi_r)

        ###############################################################    
        ### compute the CCF without defining a random catalog ###
        points_g1 = 15000
        points_g2 = 15000
        points_r  = 20000

        BoxSize = 500.0  #Mpc/h
        Rmin    = 1.0    #Mpc/h
        Rmax    = 50.0   #Mpc/h
        bins    = 30

        #get the random particles positions reading/creating a random catalogue
        pos_r,RR_name = create_random_catalogue(points_r,Rmin,Rmax,bins,BoxSize)

        D1D2_action = 'compute';  D1D2_name = 'D1D2.dat'
        D1R_action  = 'compute';  D1R_name  = 'D1R.dat'
        D2R_action  = 'compute';  D2R_name  = 'D2R.dat'
        RR_action   = 'read'  

        pos_g1 = None;  pos_g2 = None

        if myrank==0:
            pos_g1 = np.random.random((points_g1,3))*BoxSize
            pos_g2 = np.random.random((points_g2,3))*BoxSize
            pos_r  = np.random.random((points_r,3))*BoxSize

        r,xi_r = TPCCF(pos_g1,pos_g2,pos_r,BoxSize,
                       D1D2_action,D1R_action,D2R_action,RR_action,
                       D1D2_name,D1R_name,D2R_name,RR_name,
                       bins,Rmin,Rmax,verbose=True)

        if myrank == 0:
            print(r);  print(xi_r)

        ###############################################################    
        ### create/read a random catalogue ###
        points_r  = 20000

        BoxSize = 500.0  #Mpc/h
        Rmin    = 1.0    #Mpc/h
        Rmax    = 50.0   #Mpc/h
        bins    = 30

        pos_r,RR_file = create_random_catalogue(points_r,Rmin,Rmax,bins,BoxSize)

        if myrank==0:
            print('RR_file =',RR_file);  print (pos_r)


