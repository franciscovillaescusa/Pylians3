import numpy as np
import sys

def mean(a):
    return np.sum(a)*1.0/len(a)

def standard_deviation(a):
    a_mean=np.sum(a)*1.0/len(a)
    b=np.sqrt(np.sum((a-a_mean)**2)/(len(a)-1.0))
    return b

def nearest(dist,boxsize):
    if (dist>+boxsize/2.0):
        dist-=boxsize
    if (dist<-boxsize/2.0):
        dist+=boxsize
    return dist

def pdf_log10(h,interv,minimum=0.0,maximum=0.0):
    if minimum==0.0 and maximum==0.0:
        minimo=np.min(h)
        maximo=np.max(h)
    else:
        minimo=minimum
        maximo=maximum
    bins_histo=np.logspace(np.log10(minimo),np.log10(maximo),interv+1)
    hist=np.histogram(h,bins=bins_histo)[0]
    if np.sum(hist)!=len(h) and minimum==0.0 and maximum==0.0:
        print('pdf does not contain all the elements')
    pdf=[]
    for i in range(interv):
        point1=bins_histo[i]
        point2=bins_histo[i+1]
        
        #choose one of the following definitions
        #pdf.append([10**(0.5*(np.log10(point1*point2))),hist[i]*1.0/(np.log(point2/point1)),np.sqrt(hist[i])/(np.log(point2/point1))])
        pdf.append([0.5*(point1+point2),hist[i]*1.0/(np.log(point2/point1)),np.sqrt(hist[i])/(np.log(point2/point1))])
        #pdf.append([10**(0.5*(np.log10(point1*point2))),hist[i]*1.0/(point2-point1),np.sqrt(hist[i])/(point2-point1)])                               

    pdf=np.array(pdf)
    return pdf

#pos is an array of 3 components representing the coordenate of the center of a halo. R is the radius of the sphere over which we want to compute. If that sphere, centered in the halo center intersects with box boundaries, returns True, otherwise return False
def in_border(pos,R,BoxSize):
    if any(pos+R>BoxSize) or any(pos-R<0):
        return True
    else:
        return False
################################################################################
#isolation_level determines the maximum mass of halos surrounding the main halo.
#isolation_level=0.1 means that a halo is considered isolated if there are no halos within times_radius x halos_radius[halo_index] with masses larger than 0.1 x halos_mass[halo_index]
def isolated(halo_index,halos_mass,halos_pos,halos_radius,BoxSize,times_radius,isolation_level=1.0):

    if isolation_level<1.0:
        indexes=np.where(halos_mass>halos_mass[halo_index]*isolation_level)[0]
        #remove the halo itself
        id=np.where(indexes==halo_index)[0]
        indexes=np.delete(indexes,id)
    else:
        indexes=np.where(halos_mass>halos_mass[halo_index]*isolation_level)[0]
    
    if indexes!=[]:
        distances=halos_pos[indexes]-halos_pos[halo_index]

        x=distances[:,0]
        y=distances[:,1]
        z=distances[:,2]
        del distances

        beyond=np.where(x>BoxSize/2.0)
        x[beyond]-=BoxSize
        beyond=np.where(x<-BoxSize/2.0)
        x[beyond]+=BoxSize

        beyond=np.where(y>BoxSize/2.0)
        y[beyond]-=BoxSize
        beyond=np.where(y<-BoxSize/2.0)
        y[beyond]+=BoxSize

        beyond=np.where(z>BoxSize/2.0)
        z[beyond]-=BoxSize
        beyond=np.where(z<-BoxSize/2.0)
        z[beyond]+=BoxSize

        distances=np.sqrt(x**2+y**2+z**2)
    
        if np.min(distances)<times_radius*halos_radius[halo_index]:
            return False
        else:
            return True
    else:
        return True
################################################################################
class PAR_inside:
    def __init__(self,PAR_pos,BoxSize,halo_pos,halo_radius,times_radius):

        R=halo_radius*times_radius

        PAR_X=PAR_pos[:,0]-halo_pos[0]
        PAR_Y=PAR_pos[:,1]-halo_pos[1]
        PAR_Z=PAR_pos[:,2]-halo_pos[2]
        del PAR_pos

        if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
            beyond=np.where(PAR_X>BoxSize/2.0)
            PAR_X[beyond]-=BoxSize
            beyond=np.where(PAR_X<-BoxSize/2.0)
            PAR_X[beyond]+=BoxSize

            beyond=np.where(PAR_Y>BoxSize/2.0)
            PAR_Y[beyond]-=BoxSize
            beyond=np.where(PAR_Y<-BoxSize/2.0)
            PAR_Y[beyond]+=BoxSize

            beyond=np.where(PAR_Z>BoxSize/2.0)
            PAR_Z[beyond]-=BoxSize
            beyond=np.where(PAR_Z<-BoxSize/2.0)
            PAR_Z[beyond]+=BoxSize

        radius=np.sqrt(PAR_X**2+PAR_Y**2+PAR_Z**2)
        del PAR_X,PAR_Y,PAR_Z
        inside=np.where(radius<R)[0]
        self.len=len(inside)
        self.id=inside
        del radius,inside

class DM_inside:
    def __init__(self,DM_pos,BoxSize,rhocrit,Omega_DM,halo_pos,halo_virial_radius,id_number,times_virial_radius):
        self.id=id_number

        R=halo_virial_radius*times_virial_radius

        DM_X=DM_pos[:,0]-halo_pos[0]
        DM_Y=DM_pos[:,1]-halo_pos[1]
        DM_Z=DM_pos[:,2]-halo_pos[2]

        if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
            beyond=np.where(DM_X>BoxSize/2.0)
            DM_X[beyond]-=BoxSize
            beyond=np.where(DM_X<-BoxSize/2.0)
            DM_X[beyond]+=BoxSize

            beyond=np.where(DM_Y>BoxSize/2.0)
            DM_Y[beyond]-=BoxSize
            beyond=np.where(DM_Y<-BoxSize/2.0)
            DM_Y[beyond]+=BoxSize

            beyond=np.where(DM_Z>BoxSize/2.0)
            DM_Z[beyond]-=BoxSize
            beyond=np.where(DM_Z<-BoxSize/2.0)
            DM_Z[beyond]+=BoxSize

        radius=np.sqrt(DM_X**2+DM_Y**2+DM_Z**2)
        del DM_X,DM_Y,DM_Z
        inside=np.where(radius<R)[0]
        self.DM_len=len(inside)

        f=open('DMhalo'+str(id_number),'w')
        f.write (str(halo_pos[0])+' '+str(halo_pos[1])+' '+str(halo_pos[2])+' '+str(halo_virial_radius)+'\n')
        for i in range(len(inside)):
            f.write(str(DM_pos[inside[i],0])+' '+str(DM_pos[inside[i],1])+' '+str(DM_pos[inside[i],2])+'\n')
        f.close()
        del radius,inside


class PAR_cylindrical:
    #note that intervals has to be an odd number. When calling the function it is a good practice to do 2*intervals+1, where intervals are the number of bins over which the profile is computed
    def __init__(self,PAR_pos,BoxSize,halo_pos,halo_radius,h_max,V,intervals):

        PAR_pos=PAR_pos-halo_pos

        Rmax=np.sqrt(halo_radius**2+h_max**2) #max distance to halo center
        if any(halo_pos+Rmax>BoxSize) or any(halo_pos-Rmax<0.0):
            beyond=np.where(PAR_pos[:,0]>BoxSize/2.0)
            PAR_pos[beyond,0]-=BoxSize
            beyond=np.where(PAR_pos[:,0]<-BoxSize/2.0)
            PAR_pos[beyond,0]+=BoxSize

            beyond=np.where(PAR_pos[:,1]>BoxSize/2.0)
            PAR_pos[beyond,1]-=BoxSize
            beyond=np.where(PAR_pos[:,1]<-BoxSize/2.0)
            PAR_pos[beyond,1]+=BoxSize

            beyond=np.where(PAR_pos[:,2]>BoxSize/2.0)
            PAR_pos[beyond,2]-=BoxSize
            beyond=np.where(PAR_pos[:,2]<-BoxSize/2.0)
            PAR_pos[beyond,2]+=BoxSize

        #compute the minimum distance between a point and the axis V
        numerator=np.cross(PAR_pos,V)
        numerator=numerator[:,0]**2+numerator[:,1]**2+numerator[:,2]**2
        radius_h=np.sqrt(numerator/np.dot(V,V))
        del numerator
        #compute the distance between halo center and the projection of a point over the V axis
        z=np.dot(PAR_pos,V)/np.sqrt(np.dot(V,V))

        inside1=radius_h<halo_radius
        inside2=np.abs(z)<h_max
        inside=np.where(inside1*inside2==True)
        particles=PAR_pos[inside]
        z=z[inside]
        del PAR_pos,radius_h

        interv=np.linspace(-h_max,h_max,intervals)
        hist=np.histogram(z,bins=interv)[0] #the histogram is created over intervals-1 bins
        if np.sum(hist).astype(np.int64)!=len(particles):
            print('not all particles sorted')
            print(np.sum(hist),len(particles))
            sys.exit()
        del z

        length=int((intervals-1)//2)
        bin=np.empty(length,dtype=np.float32)
        values=np.empty(length,dtype=np.float32)
        errors=np.empty(length,dtype=np.float32)
        for i in range(length):
            bin[i]=abs(0.5*(interv[i]+interv[i+1]))/halo_radius
            if hist[intervals-2-i]==0 and hist[i]==0:
                values[i]=0.0
                errors[i]=0.0
            else:
                a=hist[intervals-2-i]*1.0
                b=hist[i]*1.0
                values[i]=(a-b)/(a+b)
                errors[i]=np.sqrt((1.0+values[i]**2)/(a+b))
           
        self.PAR_len=len(inside)
        self.id=inside
        self.bins=bin
        self.values=values
        self.errors=errors





    



################################################################################
class PAR_profile:
    def __init__(self,PAR_pos,PAR_nall,BoxSize,halo_index,halo_pos,halo_radius,times_radius,number_of_bins,radius_first_bin,r_bin_max,mean_R,write_file=True):

        #R=halo_radius*times_radius
        R=mean_R

        PAR_X=PAR_pos[:,0]-halo_pos[0]
        PAR_Y=PAR_pos[:,1]-halo_pos[1]
        PAR_Z=PAR_pos[:,2]-halo_pos[2]
        del PAR_pos

        if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
            beyond=np.where(PAR_X>BoxSize/2.0)
            PAR_X[beyond]-=BoxSize
            beyond=np.where(PAR_X<-BoxSize/2.0)
            PAR_X[beyond]+=BoxSize

            beyond=np.where(PAR_Y>BoxSize/2.0)
            PAR_Y[beyond]-=BoxSize
            beyond=np.where(PAR_Y<-BoxSize/2.0)
            PAR_Y[beyond]+=BoxSize

            beyond=np.where(PAR_Z>BoxSize/2.0)
            PAR_Z[beyond]-=BoxSize
            beyond=np.where(PAR_Z<-BoxSize/2.0)
            PAR_Z[beyond]+=BoxSize

        radius=np.sqrt(PAR_X**2+PAR_Y**2+PAR_Z**2)
        del PAR_X,PAR_Y,PAR_Z
        inside=np.where(radius<R)[0]
        radius_R=radius[inside]
        del inside,radius

        max_radius=np.max(radius_R)
        if (max_radius>R):
            print('error',max_radius)
            sys.exit()

        PAR_background_density=PAR_nall/BoxSize**3

        bin=np.zeros(number_of_bins,dtype=np.uint32)
        overdensity=np.empty(number_of_bins)
        #r_bin_max=np.logspace(np.log10(radius_first_bin),np.log10(R),number_of_bins)
        r_bin_min=np.empty(number_of_bins)
        r_bin_min[0]=0.0
        r_bin_min[1:number_of_bins]=r_bin_max[0:number_of_bins-1]
        r_bin=0.5*(r_bin_min+r_bin_max)
        volume_bin=4.0*np.pi*(r_bin_max**3-r_bin_min**3)/3.0
        for j in range(number_of_bins):
            len_bin=len(np.where(radius_R<=r_bin_max[j])[0])
            bin[j]=len_bin-np.sum(bin)
            overdensity[j]=bin[j]*1.0/(volume_bin[j]*PAR_background_density)

        if np.sum(bin).astype(np.uint32)!=len(radius_R):
            print('not all particles counted')
            print('counted ',np.sum(bin),' of a total of',len(radius))
            sys.exit()
        del radius_R,r_bin_max,r_bin_min,bin

        self.bin_radius=r_bin
        self.bin_overdensity=overdensity

        if write_file:
            f=open('NUprofile'+str(halo_index),'w')
            for j in range(number_of_bins):
                f.write(str(r_bin[j])+' '+str(overdensity[j])+'\n')
            f.close()
################################################################################
#compute the velocity distribution within a given CDM halo
class VEL_profile_halo:
    def __init__(self,PAR_pos,PAR_vel,BoxSize,halo_pos,halo_radius,times_radius,interv,V_min=0.0,V_max=0.0,scale='lin'):

        R=halo_radius*times_radius

        PAR_X=PAR_pos[:,0]-halo_pos[0]
        PAR_Y=PAR_pos[:,1]-halo_pos[1]
        PAR_Z=PAR_pos[:,2]-halo_pos[2]
        del PAR_pos

        if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
            beyond=np.where(PAR_X>BoxSize/2.0)
            PAR_X[beyond]-=BoxSize
            beyond=np.where(PAR_X<-BoxSize/2.0)
            PAR_X[beyond]+=BoxSize

            beyond=np.where(PAR_Y>BoxSize/2.0)
            PAR_Y[beyond]-=BoxSize
            beyond=np.where(PAR_Y<-BoxSize/2.0)
            PAR_Y[beyond]+=BoxSize

            beyond=np.where(PAR_Z>BoxSize/2.0)
            PAR_Z[beyond]-=BoxSize
            beyond=np.where(PAR_Z<-BoxSize/2.0)
            PAR_Z[beyond]+=BoxSize

        radius=np.sqrt(PAR_X**2+PAR_Y**2+PAR_Z**2)
        del PAR_X,PAR_Y,PAR_Z
        inside=np.where(radius<R)[0]

        V=np.sqrt(PAR_vel[inside,0]**2+PAR_vel[inside,1]**2+PAR_vel[inside,2]**2)
        del inside
        if V_min==0.0 and V_max==0.0:
            V_min=np.min(V)
            V_max=np.max(V)
        if scale=='lin':
            bins_histo=np.linspace(V_min,V_max,interv+1)
        elif scale=='log':
            bins_histo=np.logspace(np.log10(V_min),np.log10(V_max),interv+1)
        else:
            print('error:choose scale lin or log')
            sys.exit()

        if len(V)!=0:
            hist=np.histogram(V,bins=bins_histo)[0]*1.0/len(V)
            self.bin_h=hist
        else:
            self.bin_h=np.zeros(interv,dtype=np.float64)

        self.bin_v=bins_histo


def bulk_V(PAR_pos,PAR_vel,halo_pos,halo_radius,times_radius,BoxSize,rms=False):

    R=halo_radius*times_radius
    PAR_pos=PAR_pos-halo_pos

    #this is valid as long as R is much smaller than the BoxSize
    if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
        beyond=np.where(PAR_pos[:,0]>BoxSize/2.0)
        PAR_pos[beyond,0]-=BoxSize
        beyond=np.where(PAR_pos[:,0]<-BoxSize/2.0)
        PAR_pos[beyond,0]+=BoxSize

        beyond=np.where(PAR_pos[:,1]>BoxSize/2.0)
        PAR_pos[beyond,1]-=BoxSize
        beyond=np.where(PAR_pos[:,1]<-BoxSize/2.0)
        PAR_pos[beyond,1]+=BoxSize
        
        beyond=np.where(PAR_pos[:,2]>BoxSize/2.0)
        PAR_pos[beyond,2]-=BoxSize
        beyond=np.where(PAR_pos[:,2]<-BoxSize/2.0)
        PAR_pos[beyond,2]+=BoxSize
        del beyond

    radius=np.sqrt(PAR_pos[:,0]**2+PAR_pos[:,1]**2+PAR_pos[:,2]**2)
    del PAR_pos
    inside=np.where(radius<R)

    number_of_particles=len(inside[0])*1.0

    print('number of particles=',number_of_particles)

    max_radius=np.max(radius[inside])
    if (max_radius>R):
        print('error',max_radius)
        sys.exit()
    del radius

    bulk_vel=np.zeros(3,np.float32)
    bulk_vel[0]=np.sum(PAR_vel[inside[0],0])
    bulk_vel[1]=np.sum(PAR_vel[inside[0],1])
    bulk_vel[2]=np.sum(PAR_vel[inside[0],2])
    bulk_vel_mod=np.sqrt(bulk_vel[0]**2+bulk_vel[1]**2+bulk_vel[2]**2)
    
    velocities=PAR_vel[inside[0],0]**2+PAR_vel[inside[0],1]**2+PAR_vel[inside[0],2]**2
    del inside
    rms=np.sqrt(np.sum(velocities)/len(velocities))
    V_mean=np.sum(np.sqrt(velocities))/len(velocities)
    sigma=np.sqrt(np.sum((np.sqrt(velocities)-V_mean)**2)/len(velocities))

    return bulk_vel/number_of_particles,V_mean,rms,sigma,number_of_particles
###############################################################################
#compute the peculiar velocity of particles in shell within radius
#halo_radius x times_radius1 and halo_radius x times_radius2
#times_radius2 has to be larger than times_radius1
def bulk_shell_V(PAR_pos,PAR_vel,halo_pos,halo_radius,times_radius1,times_radius2,BoxSize):

    R2=halo_radius*times_radius2
    R1=halo_radius*times_radius1
    PAR_pos=PAR_pos-halo_pos

    #this is valid as long as R is much smaller than the BoxSize
    if any(halo_pos+R2>BoxSize) or any(halo_pos-R2<0.0):
        beyond=np.where(PAR_pos[:,0]>BoxSize/2.0)
        PAR_pos[beyond,0]-=BoxSize
        beyond=np.where(PAR_pos[:,0]<-BoxSize/2.0)
        PAR_pos[beyond,0]+=BoxSize

        beyond=np.where(PAR_pos[:,1]>BoxSize/2.0)
        PAR_pos[beyond,1]-=BoxSize
        beyond=np.where(PAR_pos[:,1]<-BoxSize/2.0)
        PAR_pos[beyond,1]+=BoxSize
        
        beyond=np.where(PAR_pos[:,2]>BoxSize/2.0)
        PAR_pos[beyond,2]-=BoxSize
        beyond=np.where(PAR_pos[:,2]<-BoxSize/2.0)
        PAR_pos[beyond,2]+=BoxSize
        del beyond

    radius=np.sqrt(PAR_pos[:,0]**2+PAR_pos[:,1]**2+PAR_pos[:,2]**2)
    del PAR_pos
    inside1=np.where(radius<R1)[0]
    inside2=np.where(radius<R2)[0]
    n_particles=len(inside2)-len(inside1)

    bulk_vel=np.zeros(3,np.float32)
    bulk_vel[0]=np.sum(PAR_vel[inside2,0])-np.sum(PAR_vel[inside1,0])
    bulk_vel[1]=np.sum(PAR_vel[inside2,1])-np.sum(PAR_vel[inside1,1])
    bulk_vel[2]=np.sum(PAR_vel[inside2,2])-np.sum(PAR_vel[inside1,2])
    del inside1,inside2

    return bulk_vel/n_particles
###############################################################################
def vel_dispersion(PAR_pos,PAR_vel,halo_pos,halo_radius,times_radius,BoxSize):

    R=halo_radius*times_radius
    PAR_pos=PAR_pos-halo_pos

    #this is valid as long as R is much smaller than the BoxSize
    if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
        beyond=np.where(PAR_pos[:,0]>BoxSize/2.0)
        PAR_pos[beyond,0]-=BoxSize
        beyond=np.where(PAR_pos[:,0]<-BoxSize/2.0)
        PAR_pos[beyond,0]+=BoxSize

        beyond=np.where(PAR_pos[:,1]>BoxSize/2.0)
        PAR_pos[beyond,1]-=BoxSize
        beyond=np.where(PAR_pos[:,1]<-BoxSize/2.0)
        PAR_pos[beyond,1]+=BoxSize
        
        beyond=np.where(PAR_pos[:,2]>BoxSize/2.0)
        PAR_pos[beyond,2]-=BoxSize
        beyond=np.where(PAR_pos[:,2]<-BoxSize/2.0)
        PAR_pos[beyond,2]+=BoxSize
        del beyond

    radius=np.sqrt(PAR_pos[:,0]**2+PAR_pos[:,1]**2+PAR_pos[:,2]**2)
    del PAR_pos
    inside=np.where(radius<R)[0]

    number_of_particles=len(inside)*1.0

    max_radius=np.max(radius[inside])
    if (max_radius>R):
        print('error',max_radius)
        sys.exit()
    del radius

    bulk_vel=np.zeros(3,np.float32)
    bulk_vel[0]=np.sum(PAR_vel[inside,0])
    bulk_vel[1]=np.sum(PAR_vel[inside,1])
    bulk_vel[2]=np.sum(PAR_vel[inside,2])
    bulk_vel=bulk_vel/number_of_particles

    vel=PAR_vel[inside]-bulk_vel
    dispersion=np.sum(vel[:,0]**2+vel[:,1]**2+vel[:,2]**2)/(3.0*len(vel))
    dispersion=np.sqrt(dispersion)

    return dispersion
###############################################################################
def low_V(PAR_pos,PAR_vel,halo_pos,halo_radius,times_radius,BoxSize):

    R=halo_radius*times_radius
    PAR_pos=PAR_pos-halo_pos

    #this is valid as long as R is much smaller than the BoxSize
    if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
        beyond=np.where(PAR_pos[:,0]>BoxSize/2.0)
        PAR_pos[beyond,0]-=BoxSize
        beyond=np.where(PAR_pos[:,0]<-BoxSize/2.0)
        PAR_pos[beyond,0]+=BoxSize

        beyond=np.where(PAR_pos[:,1]>BoxSize/2.0)
        PAR_pos[beyond,1]-=BoxSize
        beyond=np.where(PAR_pos[:,1]<-BoxSize/2.0)
        PAR_pos[beyond,1]+=BoxSize
        
        beyond=np.where(PAR_pos[:,2]>BoxSize/2.0)
        PAR_pos[beyond,2]-=BoxSize
        beyond=np.where(PAR_pos[:,2]<-BoxSize/2.0)
        PAR_pos[beyond,2]+=BoxSize
        del beyond

    radius=np.sqrt(PAR_pos[:,0]**2+PAR_pos[:,1]**2+PAR_pos[:,2]**2)
    del PAR_pos
    inside=np.where(radius<R)[0]

    number_of_particles=len(inside)
    print('number of particles=',number_of_particles)

    max_radius=np.max(radius[inside])
    if (max_radius>R):
        print('error',max_radius)
        sys.exit()
    del radius

    vel=PAR_vel[inside]
    mod_vel=np.sqrt(vel[:,0]**2+vel[:,1]**2+vel[:,2]**2)
    low=np.where(mod_vel<400.0)[0]

    print(low)
    return inside,low
    


################################################################################
#this routine returns the position of the center of mass of a given particle distribution
def COM(PAR_pos,halo_pos,halo_radius,times_radius,BoxSize):

    R=halo_radius*times_radius
    PAR_pos=PAR_pos-halo_pos

    #this is valid as long as R is much smaller than the BoxSize
    if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
        beyond=np.where(PAR_pos[:,0]>BoxSize/2.0)
        PAR_pos[beyond,0]-=BoxSize
        beyond=np.where(PAR_pos[:,0]<-BoxSize/2.0)
        PAR_pos[beyond,0]+=BoxSize

        beyond=np.where(PAR_pos[:,1]>BoxSize/2.0)
        PAR_pos[beyond,1]-=BoxSize
        beyond=np.where(PAR_pos[:,1]<-BoxSize/2.0)
        PAR_pos[beyond,1]+=BoxSize
        
        beyond=np.where(PAR_pos[:,2]>BoxSize/2.0)
        PAR_pos[beyond,2]-=BoxSize
        beyond=np.where(PAR_pos[:,2]<-BoxSize/2.0)
        PAR_pos[beyond,2]+=BoxSize
        del beyond

    radius=np.sqrt(PAR_pos[:,0]**2+PAR_pos[:,1]**2+PAR_pos[:,2]**2)
    inside=np.where(radius<R)

    max_radius=np.max(radius[inside])
    if (max_radius>R):
        print('error',max_radius)
        sys.exit()
    del radius

    COM=np.zeros(3,np.float64)
    COM[0]=np.sum(PAR_pos[inside[0],0],dtype=np.float64)
    COM[1]=np.sum(PAR_pos[inside[0],1],dtype=np.float64)
    COM[2]=np.sum(PAR_pos[inside[0],2],dtype=np.float64)
    del PAR_pos

    return COM/len(inside[0])
###############################################################################

################################################################################
#this routine returns the number of particles within a given radius from a point
def interior_mass(PAR_pos,halo_pos,halo_radius,times_radius,BoxSize):

    R=halo_radius*times_radius

    PAR_X=PAR_pos[:,0]-halo_pos[0]
    PAR_Y=PAR_pos[:,1]-halo_pos[1]
    PAR_Z=PAR_pos[:,2]-halo_pos[2]
    del PAR_pos

    #this is valid as long as R is much smaller than the BoxSize
    if any(halo_pos+R>BoxSize) or any(halo_pos-R<0.0):
        beyond=np.where(PAR_X>BoxSize/2.0)
        PAR_X[beyond]-=BoxSize
        beyond=np.where(PAR_X<-BoxSize/2.0)
        PAR_X[beyond]+=BoxSize

        beyond=np.where(PAR_Y>BoxSize/2.0)
        PAR_Y[beyond]-=BoxSize
        beyond=np.where(PAR_Y<-BoxSize/2.0)
        PAR_Y[beyond]+=BoxSize
        
        beyond=np.where(PAR_Z>BoxSize/2.0)
        PAR_Z[beyond]-=BoxSize
        beyond=np.where(PAR_Z<-BoxSize/2.0)
        PAR_Z[beyond]+=BoxSize
        del beyond

    radius=np.sqrt(PAR_X**2+PAR_Y**2+PAR_Z**2)
    del PAR_X,PAR_Y,PAR_Z
    inside=np.where(radius<=R)
    number_of_particles=len(inside[0])*1.0

    if number_of_particles>0.0:
        max_radius=np.max(radius[inside])
        if (max_radius>R):
            print('error',max_radius)
            sys.exit()
    del radius

    return number_of_particles













