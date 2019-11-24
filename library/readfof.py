#model read_FoF"""
#readfof(Base,snapnum,endian=None)
#Read FoF files from Gadget, P-FoF
     #Parameters:
        #basedir: where your FoF folder located
        #snapnum: snapshot number
        #long_ids: whether particles ids are uint32 or uint64
        #swap: False or True
     #return structures:
        #TotNgroups,TotNids,GroupLen,GroupOffset,GroupMass,GroupPos,GroupIDs...
     #Example:
        #--------
        #FoF_halos=readfof("/data1/villa/b500p512nu0.6z99tree",17,long_ids=True,swap=False)
        #Masses=FoF_halos.GroupMass
        #IDs=FoF_halos.GroupIDs
        #--------
        #updated time 19 Oct 2012 by wgcui

#For simulations with SFR, set SFR=True
#The physical velocities of the halos are found multiplying the field 
#GroupVel by (1+z)

import numpy as np
import os, sys
from struct import unpack

class FoF_catalog:
    def __init__(self, basedir, snapnum, long_ids=False, swap=False,
                 SFR=False, read_IDs=True, prefix='/groups_'):

        if long_ids:  format = np.uint64
        else:         format = np.uint32

        exts=('000'+str(snapnum))[-3:]

        #################  READ TAB FILES ################# 
        fnb, skip, Final = 0, 0, False
        dt1 = np.dtype((np.float32,3))
        dt2 = np.dtype((np.float32,6))
        prefix = basedir + prefix + exts + "/group_tab_" + exts + "."
        while not(Final):
            f=open(prefix+str(fnb), 'rb')
            self.Ngroups    = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.TotNgroups = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.Nids       = np.fromfile(f, dtype=np.int32,  count=1)[0]
            self.TotNids    = np.fromfile(f, dtype=np.uint64, count=1)[0]
            self.Nfiles     = np.fromfile(f, dtype=np.uint32, count=1)[0]

            TNG, NG = self.TotNgroups, self.Ngroups
            if fnb == 0:
                self.GroupLen    = np.empty(TNG, dtype=np.int32)
                self.GroupOffset = np.empty(TNG, dtype=np.int32)
                self.GroupMass   = np.empty(TNG, dtype=np.float32)
                self.GroupPos    = np.empty(TNG, dtype=dt1)
                self.GroupVel    = np.empty(TNG, dtype=dt1)
                self.GroupTLen   = np.empty(TNG, dtype=dt2)
                self.GroupTMass  = np.empty(TNG, dtype=dt2)
                if SFR:  self.GroupSFR = np.empty(TNG, dtype=np.float32)
                    
            if NG>0:
                locs=slice(skip,skip+NG)
                self.GroupLen[locs]    = np.fromfile(f,dtype=np.int32,count=NG)
                self.GroupOffset[locs] = np.fromfile(f,dtype=np.int32,count=NG)
                self.GroupMass[locs]   = np.fromfile(f,dtype=np.float32,count=NG)
                self.GroupPos[locs]    = np.fromfile(f,dtype=dt1,count=NG)
                self.GroupVel[locs]    = np.fromfile(f,dtype=dt1,count=NG)
                self.GroupTLen[locs]   = np.fromfile(f,dtype=dt2,count=NG)
                self.GroupTMass[locs]  = np.fromfile(f,dtype=dt2,count=NG)
                if SFR:
                    self.GroupSFR[locs]=np.fromfile(f,dtype=np.float32,count=NG)
                skip+=NG

                if swap:
                    self.GroupLen.byteswap(True)
                    self.GroupOffset.byteswap(True)
                    self.GroupMass.byteswap(True)
                    self.GroupPos.byteswap(True)
                    self.GroupVel.byteswap(True)
                    self.GroupTLen.byteswap(True)
                    self.GroupTMass.byteswap(True)
                    if SFR:  self.GroupSFR.byteswap(True)
                        
            curpos = f.tell()
            f.seek(0,os.SEEK_END)
            if curpos != f.tell():
                raise Exception("Warning: finished reading before EOF for tab file",fnb)
            f.close()
            fnb+=1
            if fnb==self.Nfiles: Final=True


        #################  READ IDS FILES ################# 
        if read_IDs:

            fnb,skip=0,0
            Final=False
            while not(Final):
                fname=basedir+"/groups_" + exts +"/group_ids_"+exts +"."+str(fnb)
                f=open(fname,'rb')
                Ngroups     = np.fromfile(f,dtype=np.uint32,count=1)[0]
                TotNgroups  = np.fromfile(f,dtype=np.uint32,count=1)[0]
                Nids        = np.fromfile(f,dtype=np.uint32,count=1)[0]
                TotNids     = np.fromfile(f,dtype=np.uint64,count=1)[0]
                Nfiles      = np.fromfile(f,dtype=np.uint32,count=1)[0]
                Send_offset = np.fromfile(f,dtype=np.uint32,count=1)[0]
                if fnb==0:
                    self.GroupIDs=np.zeros(dtype=format,shape=TotNids)
                if Ngroups>0:
                    if long_ids:
                        IDs=np.fromfile(f,dtype=np.uint64,count=Nids)
                    else:
                        IDs=np.fromfile(f,dtype=np.uint32,count=Nids)
                    if swap:
                        IDs=IDs.byteswap(True)
                    self.GroupIDs[skip:skip+Nids]=IDs[:]
                    skip+=Nids
                curpos = f.tell()
                f.seek(0,os.SEEK_END)
                if curpos != f.tell():
                    raise Exception("Warning: finished reading before EOF for IDs file",fnb)
                f.close()
                fnb+=1
                if fnb==Nfiles: Final=True


# This function is used to write one single file for the FoF instead of having
# many files. This will make faster the reading of the FoF file
def writeFoFCatalog(fc, tabFile, idsFile=None):
    if fc.TotNids > (1<<32)-1: raise Exception('TotNids overflow')

    f = open(tabFile, 'wb')
    np.asarray(fc.TotNgroups).tofile(f)
    np.asarray(fc.TotNgroups).tofile(f)
    np.asarray(fc.TotNids, dtype=np.int32).tofile(f)
    np.asarray(fc.TotNids).tofile(f)
    np.asarray(1, dtype=np.uint32).tofile(f)
    fc.GroupLen.tofile(f)
    fc.GroupOffset.tofile(f)
    fc.GroupMass.tofile(f)
    fc.GroupPos.tofile(f)
    fc.GroupVel.tofile(f)
    fc.GroupTLen.tofile(f)
    fc.GroupTMass.tofile(f)
    if hasattr(fc, 'GroupSFR'):
        fc.GroupSFR.tofile(f)
    f.close()

    if idsFile:
        f = open(idsFile, 'wb')
        np.asarray(fc.TotNgroups).tofile(f)
        np.asarray(fc.TotNgroups).tofile(f)
        np.asarray(fc.TotNids, dtype=np.uint32).tofile(f) 
        np.asarray(fc.TotNids).tofile(f)
        np.asarray(1, dtype=np.uint32).tofile(f)
        np.asarray(0, dtype=np.uint32).tofile(f) 
        fc.GroupIDs.tofile(f)
        f.close()



# This is an example on how to change files
"""
root = '/mnt/xfs1/home/fvillaescusa/data/Neutrino_simulations/Sims_Dec16_2/'
################################## INPUT ######################################
folders = ['0.0eV/','0.06eV/','0.10eV/','0.10eV_degenerate/',
           '0.15eV/','0.6eV/',
           '0.0eV_0.798/','0.0eV_0.807/','0.0eV_0.818/','0.0eV_0.822/',
           '0.0eV_s8c/','0.0eV_s8m/']
###############################################################################

# do a loop over the different cosmologies
for folder in folders:

    # do a loop over the different realizations
    for i in range(1,101):

        snapdir = root + folder + '%d/'%i
        
        # do a loop over the different redshift
        for snapnum in [0,1,2,3]:

            FoF_folder     = snapdir+'groups_%03d'%snapnum
            old_FoF_folder = snapdir+'original_groups_%03d'%snapnum
            if os.path.exists(FoF_folder):
                print('%s\t%d\t%d\texists'%(folder,i,snapnum))

                if os.path.exists(old_FoF_folder):
                    continue

                # create new FoF file
                f_tab = '%s/group_tab_%03d.0'%(snapdir,snapnum)
                f_ids = '%s/group_ids_%03d.0'%(snapdir,snapnum)
                FoF = readfof.FoF_catalog(snapdir,snapnum,long_ids=False,
                                          swap=False,SFR=False)
                writeFoFCatalog(FoF, f_tab, idsFile=f_ids)
           
                # rename FoF folder, create new FoF folder and move files to it
                os.system('mv '+FoF_folder+' '+old_FoF_folder)
                os.system('mkdir '+FoF_folder)
                os.system('mv '+f_tab+' '+f_ids+' '+FoF_folder)
"""
