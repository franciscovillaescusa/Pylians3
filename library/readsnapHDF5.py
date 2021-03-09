# import readsnapHDF5 as rs
# header = rs.snapshot_header("snap_063.0") 
# mass = rs.read_block("snap_063","MASS",parttype=5) # reads mass for particles of type 5

import numpy as np
import os
import sys
import math
import tables

############ 
#DATABLOCKS#
############
#descriptions of all datablocks -> add datablocks here!
#TAG:[HDF5_NAME,DIM]
datablocks = {"POS ":["Coordinates",3], 
              "VEL ":["Velocities",3],
              "ID  ":["ParticleIDs",1],
              "MASS":["Masses",1],
              "U   ":["InternalEnergy",1],
              "RHO ":["Density",1],
              "VOL ":["Volume",1],
              "CMCE":["Center-of-Mass",3],
              "AREA":["Surface Area",1],
              "NFAC":["Number of faces of cell",1],
              "NE  ":["ElectronAbundance",1],
              "NH  ":["NeutralHydrogenAbundance",1],
              "HSML":["SmoothingLength",1],
              "SFR ":["StarFormationRate",1],
              "AGE ":["StellarFormationTime",1],
              "Z   ":["Metallicity",1],
              "ACCE":["Acceleration",3],
              "VEVE":["VertexVelocity",3],
              "FACA":["MaxFaceAngle",1],
              "COOR":["CoolingRate",1],
              "POT ":["Potential",1],
              "MACH":["MachNumber",1],
              "GAGE":["GFM StellarFormationTime",1],
              "GIMA":["GFM InitialMass",1],
              "GZ  ":["GFM Metallicity",1],
              "GMET":["GFM Metals",9],
              "GMRE":["GFM MetalsReleased",9],
              "GMAR":["GFM MetalMassReleased", 1]}


#####################################################################################################################
#                                                    READING ROUTINES			                            #
#####################################################################################################################


########################### 
#CLASS FOR SNAPSHOT HEADER#
###########################  
class snapshot_header:
    def __init__(self, *args, **kwargs):
        if (len(args) == 1):
            filename = args[0]
            if os.path.exists(filename):
                curfilename=filename
            elif os.path.exists(filename+".hdf5"):
                curfilename = filename+".hdf5"
            elif os.path.exists(filename+".0.hdf5"): 
                curfilename = filename+".0.hdf5"
            else:
                print("[error] file not found : ", filename)
                sys.exit()

            f=tables.open_file(curfilename)
            self.npart = f.root.Header._v_attrs.NumPart_ThisFile 
            self.nall = f.root.Header._v_attrs.NumPart_Total
            self.nall_highword = f.root.Header._v_attrs.NumPart_Total_HighWord
            self.massarr = f.root.Header._v_attrs.MassTable 
            self.time = f.root.Header._v_attrs.Time 
            self.redshift = f.root.Header._v_attrs.Redshift 
            self.boxsize = f.root.Header._v_attrs.BoxSize
            self.filenum = f.root.Header._v_attrs.NumFilesPerSnapshot
            self.omega0 = f.root.Header._v_attrs.Omega0
            self.omegaL = f.root.Header._v_attrs.OmegaLambda
            self.hubble = f.root.Header._v_attrs.HubbleParam
            self.sfr = f.root.Header._v_attrs.Flag_Sfr 
            self.cooling = f.root.Header._v_attrs.Flag_Cooling
            self.stellar_age = f.root.Header._v_attrs.Flag_StellarAge
            self.metals = f.root.Header._v_attrs.Flag_Metals
            self.feedback = f.root.Header._v_attrs.Flag_Feedback
            self.double = f.root.Header._v_attrs.Flag_DoublePrecision #GADGET-2
            f.close()

        else:
            #read arguments
            self.npart = kwargs.get("npart")
            self.nall = kwargs.get("nall")
            self.nall_highword = kwargs.get("nall_highword")
            self.massarr = kwargs.get("massarr")
            self.time = kwargs.get("time")
            self.redshift = kwargs.get("redshift")
            self.boxsize = kwargs.get("boxsize")
            self.filenum = kwargs.get("filenum")
            self.omega0 = kwargs.get("omega0")
            self.omegaL = kwargs.get("omegaL")
            self.hubble = kwargs.get("hubble")
            self.sfr = kwargs.get("sfr")
            self.cooling = kwargs.get("cooling")
            self.stellar_age = kwargs.get("stellar_age")
            self.metals = kwargs.get("metals")
            self.feedback = kwargs.get("feedback")
            self.double = kwargs.get("double")

            #set default values
            if (self.npart == None):
                self.npart = np.array([0,0,0,0,0,0], dtype="int32")
            if (self.nall == None):
                self.nall  = np.array([0,0,0,0,0,0], dtype="uint32")
            if (self.nall_highword == None):				
                self.nall_highword = np.array([0,0,0,0,0,0], dtype="uint32")
            if (self.massarr == None):
                self.massarr = np.array([0,0,0,0,0,0], dtype="float64")
            if (self.time == None):				
                self.time = np.array([0], dtype="float64")
            if (self.redshift == None):				
                self.redshift = np.array([0], dtype="float64")
            if (self.boxsize == None):				
                self.boxsize = np.array([0], dtype="float64")
            if (self.filenum == None):
                self.filenum = np.array([1], dtype="int32")
            if (self.omega0 == None):
                self.omega0 = np.array([0], dtype="float64")
            if (self.omegaL == None):
                self.omegaL = np.array([0], dtype="float64")
            if (self.hubble == None):
                self.hubble = np.array([0], dtype="float64")
            if (self.sfr == None):	
                self.sfr = np.array([0], dtype="int32")            
            if (self.cooling == None):	
                self.cooling = np.array([0], dtype="int32")
            if (self.stellar_age == None):	
                self.stellar_age = np.array([0], dtype="int32")
            if (self.metals == None):	
                self.metals = np.array([0], dtype="int32")
            if (self.feedback == None):	
                self.feedback = np.array([0], dtype="int32")
            if (self.double == None):
                self.double = np.array([0], dtype="int32")
        
            
        

##############################
#READ ROUTINE FOR SINGLE FILE#
############################## 
def read_block_single_file(filename, block_name, dim2, parttype=-1, no_mass_replicate=False, verbose=False):

  if (verbose):
      print("[single] reading file           : ", filename)   	
      print("[single] reading                : ", block_name)
      
  head = snapshot_header(filename)
  npart = head.npart
  massarr = head.massarr
  nall = head.nall
  filenum = head.filenum
  doubleflag = head.double #GADGET-2
  #doubleflag = 0 #GADGET-2
  del head

  f=tables.open_file(filename)


  #read specific particle type 
  if parttype>=0:
    if (verbose):
        print("[single] parttype               : ", parttype)
    if ((block_name=="Masses") & (npart[parttype]>0) & (massarr[parttype]>0)):
        if (verbose):
            print("[single] replicate mass block")
        ret_val=np.repeat(massarr[parttype], npart[parttype])
    else:
        part_name='PartType'+str(parttype)
        ret_val = f.root._f_get_child(part_name)._f_get_child(block_name)[:]
    if (verbose):
        print("[single] read particles (total) : ", ret_val.shape[0]//dim2)

  #read all particle types
  if parttype==-1:
    first=True
    dim1=0
    for parttype in range(0,5):
        part_name='PartType'+str(parttype)
        if (f.root.__contains__(part_name)):
            if (verbose):
                print("[single] parttype               : ", parttype) 
                print("[single] massarr                : ", massarr)
                print("[single] npart                  : ", npart)

            if ((block_name=="Masses") & (npart[parttype]>0) & (massarr[parttype]>0) & (no_mass_replicate==False)):
                if (verbose):
                    print("[single] replicate mass block")
                if (first):
                    data=np.repeat(massarr[parttype], npart[parttype])
                    dim1+=data.shape[0]
                    ret_val=data
                    first=False
                else:
                    data=np.repeat(massarr[parttype], npart[parttype])
                    dim1+=data.shape[0]
                    ret_val=np.append(ret_val, data)
                if (verbose):
                    print("[single] read particles (total) : ", ret_val.shape[0]//dim2)
                if (doubleflag==0):
                    ret_val=ret_val.astype("float32")
            if (f.root._f_get_child(part_name).__contains__(block_name)):
                if (first):
                    data=f.root._f_get_child(part_name)._f_get_child(block_name)[:]
                    dim1+=data.shape[0]
                    ret_val=data
                    first=False
                else:
                    data=f.root._f_get_child(part_name)._f_get_child(block_name)[:]
                    dim1+=data.shape[0]
                    ret_val=np.append(ret_val, data)
                if (verbose):
                    print("[single] read particles (total) : ", ret_val.shape[0]//dim2)

    if ((dim1>0) & (dim2>1)):
        ret_val=ret_val.reshape(dim1,dim2)

  f.close()

  return ret_val

##############
#READ ROUTINE#
##############
def read_block(filename, block, parttype=-1, no_mass_replicate=False, verbose=False):
  if (verbose):
          print("reading block          : ", block)

  if parttype not in [-1,0,1,2,3,4,5]:
    print("[error] wrong parttype given")
    sys.exit()

  curfilename=filename+".hdf5"

  if os.path.exists(curfilename):
    multiple_files=False
  elif os.path.exists(filename+".0"+".hdf5"):
    curfilename = filename+".0"+".hdf5"
    multiple_files=True
  else:
    print("[error] file not found : ", filename)
    sys.exit()

  head = snapshot_header(curfilename)
  filenum = head.filenum
  del head

  #if (datablocks.has_key(block)):
  if block in datablocks:
        block_name=datablocks[block][0]
        dim2=datablocks[block][1]
        first=True
        if (verbose):
                print("Reading HDF5           : ", block_name)
                print("Data dimension         : ", dim2)
                print("Multiple file          : ", multiple_files)
  else:
        print("[error] Block type ", block, "not known!")
        sys.exit()


  if (multiple_files):
    first=True
    dim1=0
    for num in range(0,filenum):
        curfilename=filename+"."+str(num)+".hdf5"
        if (verbose):
            print("Reading file           : ", num, curfilename)
        if (first):
            data = read_block_single_file(curfilename, block_name, dim2, parttype, verbose)
            dim1+=data.shape[0]
            ret_val = data
            first = False 
        else:
            data = read_block_single_file(curfilename, block_name, dim2, parttype, verbose)
            dim1+=data.shape[0]
            ret_val=np.append(ret_val, data)
        if (verbose):
            print("Read particles (total) : ", ret_val.shape[0]//dim2)

    if ((dim1>0) & (dim2>1)):
        ret_val=ret_val.reshape(dim1,dim2)	
  else:
    ret_val=read_block_single_file(curfilename, block_name, dim2, parttype, no_mass_replicate, verbose)

  return ret_val


#############
#LIST BLOCKS#
#############
def list_blocks(filename, parttype=-1, verbose=False):
  
  f=tables.open_file(filename)
  for parttype in range(0,5):
    part_name='PartType'+str(parttype)
    if (f.root.__contains__(part_name)):
        print("Parttype contains : ", parttype)
        print("-------------------")
        iter = it=datablocks.__iter__()
        next = iter.next()
        while (1):
            if (verbose):
                print("check ", next, datablocks[next][0])
            if (f.root._f_get_child(part_name).__contains__(datablocks[next][0])):
                print(next, datablocks[next][0])
            try:
                next=iter.next()
            except StopIteration:
                break
  f.close()

#################
#CONTAINS BLOCKS#
#################
def contains_block(filename, tag, parttype=-1, verbose=False):
  
  contains_flag=False
  f=tables.open_file(filename)
  for parttype in range(0,5):
        part_name='PartType'+str(parttype)
        if (f.root.__contains__(part_name)):
                iter = it=datablocks.__iter__()
                next = iter.next()
                while (1):
                        if (verbose):
                                print("check ", next, datablocks[next][0])
                        if (f.root._f_get_child(part_name).__contains__(datablocks[next][0])):
                            if (next.find(tag)>-1):
                                contains_flag=True
                        try:
                            next=iter.next()
                        except StopIteration:
                            break
  f.close() 
  return contains_flag

############
#CHECK FILE#
############
def check_file(filename):
  f=tables.open_file(filename)
  f.close()
                                                                                                                                                  






#####################################################################################################################
#                                                    WRITING ROUTINES    		                            #
#####################################################################################################################



#######################
#OPEN FILE FOR WRITING#
#######################
def openfile(filename):
    f=tables.open_file(filename, mode = "w")
    return f

############
#CLOSE FILE#
############
def closefile(f):
    f.close()

##############################
#WRITE SNAPSHOT HEADER OBJECT#
##############################
def writeheader(f, header):	
    group_header=f.createGroup(f.root, "Header")
    group_header._v_attrs.NumPart_ThisFile=header.npart
    group_header._v_attrs.NumPart_Total=header.nall
    group_header._v_attrs.NumPart_Total_HighWord=header.nall_highword
    group_header._v_attrs.MassTable=header.massarr
    group_header._v_attrs.Time=header.time
    group_header._v_attrs.Redshift=header.redshift
    group_header._v_attrs.BoxSize=header.boxsize
    group_header._v_attrs.NumFilesPerSnapshot=header.filenum
    group_header._v_attrs.Omega0=header.omega0
    group_header._v_attrs.OmegaLambda=header.omegaL
    group_header._v_attrs.HubbleParam=header.hubble
    group_header._v_attrs.Flag_Sfr=header.sfr
    group_header._v_attrs.Flag_Cooling=header.cooling
    group_header._v_attrs.Flag_StellarAge=header.stellar_age
    group_header._v_attrs.Flag_Metals=header.metals
    group_header._v_attrs.Flag_Feedback=header.feedback
    group_header._v_attrs.Flag_DoublePrecision=header.double

###############
#WRITE ROUTINE#
###############
def write_block(f, block, parttype, data):
    part_name="PartType"+str(parttype)
    if (f.root.__contains__(part_name)==False):
            group=f.createGroup(f.root, part_name)
    else:
        group=f.root._f_get_child(part_name)
    
    #if (datablocks.has_key(block)):
    if block in datablocks:
        block_name=datablocks[block][0]
        dim2=datablocks[block][1]
        if (group.__contains__(block_name)==False):
            table=f.createArray(group, block_name, data)
        else:
            print("I/O block already written")
    else:
        print("Unknown I/O block")
