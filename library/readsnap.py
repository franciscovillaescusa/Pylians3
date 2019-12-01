# routines for reading headers and data blocks from Gadget snapshot files
# usage e.g.:
#
# import readsnap as rs
# header = rs.snapshot_header("snap_063.0") # reads snapshot header
# print header.massarr
# mass = rs.read_block("snap_063","MASS",parttype=5) # reads mass for particles of type 5, using block names should work for both format 1 and 2 snapshots
# print "mass for", mass.size, "particles read"
# print mass[0:10]
#
# before using read_block, make sure that the description (and order if using format 1 snapshot files) of the data blocks
# is correct for your configuration of Gadget 
#
# for mutliple file snapshots give e.g. the filename "snap_063" rather than "snap_063.0" to read_block
# for snapshot_header the file number should be included, e.g."snap_063.0", as the headers of the files differ
#
# the returned data block is ordered by particle species even when read from a multiple file snapshot

import numpy as np
import os
import sys
import math
  
# ----- class for snapshot header ----- 

class snapshot_header:
  def __init__(self, filename):

    if os.path.exists(filename):
      curfilename = filename
    elif os.path.exists(filename+".0"):
      curfilename = filename+".0"
    else:
      print("file not found:", filename)
      sys.exit()
      
    self.filename = filename  
    f = open(curfilename,'rb')    
    blocksize = np.fromfile(f,dtype=np.int32,count=1)
    if blocksize[0] == 8:
      swap = 0
      format = 2
    elif blocksize[0] == 256:
      swap = 0
      format = 1  
    else:
      blocksize.byteswap(True)
      if blocksize[0] == 8:
        swap = 1
        format = 2
      elif blocksize[0] == 256:
        swap = 1
        format = 1
      else:
        print("incorrect file format encountered when reading header of", filename)
        sys.exit()
    
    self.format = format
    self.swap = swap
    
    if format==2:
      f.seek(16, os.SEEK_CUR)
    
    self.npart = np.fromfile(f,dtype=np.int32,count=6)
    self.massarr = np.fromfile(f,dtype=np.float64,count=6)
    self.time = (np.fromfile(f,dtype=np.float64,count=1))[0]
    self.redshift = (np.fromfile(f,dtype=np.float64,count=1))[0]
    self.sfr = (np.fromfile(f,dtype=np.int32,count=1))[0]
    self.feedback = (np.fromfile(f,dtype=np.int32,count=1))[0]
    self.nall = np.fromfile(f,dtype=np.uint32,count=6)
    self.cooling = (np.fromfile(f,dtype=np.int32,count=1))[0]
    self.filenum = (np.fromfile(f,dtype=np.int32,count=1))[0]
    self.boxsize = (np.fromfile(f,dtype=np.float64,count=1))[0]
    self.omega_m = (np.fromfile(f,dtype=np.float64,count=1))[0]
    self.omega_l = (np.fromfile(f,dtype=np.float64,count=1))[0]
    self.hubble = (np.fromfile(f,dtype=np.float64,count=1))[0]
    
    if swap:
      self.npart.byteswap(True)
      self.massarr.byteswap(True)
      self.time = self.time.byteswap()
      self.redshift = self.redshift.byteswap()
      self.sfr = self.sfr.byteswap()
      self.feedback = self.feedback.byteswap()
      self.nall.byteswap(True)
      self.cooling = self.cooling.byteswap()
      self.filenum = self.filenum.byteswap()
      self.boxsize = self.boxsize.byteswap()
      self.omega_m = self.omega_m.byteswap()
      self.omega_l = self.omega_l.byteswap()
      self.hubble = self.hubble.byteswap()
     
    f.close()
 
# ----- find offset and size of data block ----- 

def find_block(filename, format, swap, block, block_num, only_list_blocks=False):
  if (not os.path.exists(filename)):
      print("file not found:", filename)
      sys.exit()
            
  f = open(filename,'rb')
  f.seek(0, os.SEEK_END)
  filesize = f.tell()
  f.seek(0, os.SEEK_SET)
  
  found = False
  curblock_num = 1
  while ((not found) and (f.tell()<filesize)):
    if format==2:
      f.seek(4, os.SEEK_CUR)
      curblock = f.read(4).decode()
      if (block == curblock):
        found = True
      f.seek(8, os.SEEK_CUR)  
    else:
      if curblock_num==block_num:
        found = True
        
    curblocksize = (np.fromfile(f,dtype=np.uint32,count=1))[0]
    if swap:
      curblocksize = curblocksize.byteswap()
    
    # - print some debug info about found data blocks -
    #if format==2:
    #  print curblock, curblock_num, curblocksize
    #else:
    #  print curblock_num, curblocksize
    
    if only_list_blocks:
      if format==2:
        print(curblock_num,curblock,f.tell(),curblocksize)
      else:
        print(curblock_num,f.tell(),curblocksize)
      found = False
        
    
    if found:
      blocksize = curblocksize
      offset = f.tell()
    else:
      f.seek(curblocksize, os.SEEK_CUR)
      blocksize_check = (np.fromfile(f,dtype=np.uint32,count=1))[0]
      if swap: blocksize_check = blocksize_check.byteswap()
      if (curblocksize != blocksize_check):
        print("something wrong")
        sys.exit()
      curblock_num += 1
  f.close()
      
  if ((not found) and (not only_list_blocks)):
    print("Error: block not found")
    sys.exit()
    
  if (not only_list_blocks):
    return offset,blocksize
 
# ----- read data block -----
#for snapshots with very very large number of particles set nall manually
#for instance nall=np.array([0,2048**3,0,0,0,0]) 
def read_block(filename, block, parttype=-1, physical_velocities=True, 
    arepo=0, no_masses=False, verbose=False, nall=[0,0,0,0,0,0]):
  
  if (verbose):  print("reading block", block)
  
  blockadd=0
  blocksub=0
  
  if arepo==0:
    if (verbose):  print("Gadget format")
    blockadd=0
  if arepo==1:
    if (verbose):  print("Arepo format")
    blockadd=1  
  if arepo==2:
    if (verbose):  print("Arepo extended format")
    blockadd=4  
  if no_masses==True:
    if (verbose):  print("No mass block present")   
    blocksub=1
         
  if parttype not in [-1,0,1,2,3,4,5]:
    print("wrong parttype given");  sys.exit()
  
  if os.path.exists(filename):        
    curfilename = filename
    single_file = True
  elif os.path.exists(filename+".0"):   
    curfilename = filename+".0"
    single_file = False
  else:
    print("file not found:", filename)
    print("and:", curfilename);  sys.exit()
  
  head = snapshot_header(curfilename)
  format   = head.format
  swap     = head.swap
  npart    = head.npart
  massarr  = head.massarr
  filenum  = head.filenum
  redshift = head.redshift
  time     = head.time
  if np.all(nall==[0,0,0,0,0,0]):
    nall = head.nall
  if verbose:  print("FORMAT=", format)
  del head
  
  # - description of data blocks -
  # add or change blocks as needed for your Gadget version
  data_for_type = np.zeros(6,bool) # should be set to "True" below for the species for which data is stored in the data block #by doing this, the default value is False data_for_type=[False,False,False,False,False,False]
  dt = np.float32 # data type of the data in the block
  if block=="POS ":
    data_for_type[:] = True
    dt = np.dtype((np.float32,3))
    block_num = 2
  elif block=="VEL ":
    data_for_type[:] = True
    dt = np.dtype((np.float32,3))
    block_num = 3
  elif block=="ID  ":
    data_for_type[:] = True
    dt = np.uint32
    block_num = 4
#only used for format I, when file structure is HEAD,POS,VEL,ID,ACCE
  elif block=="ACCE":              #This is only for the PIETRONI project
    data_for_type[:] = True        #This is only for the PIETRONI project
    dt = np.dtype((np.float32,3))  #This is only for the PIETRONI project
    block_num = 5                  #This is only for the PIETRONI project
  elif block=="MASS":
    data_for_type[np.where(massarr==0)] = True
    block_num = 5
    if parttype>=0 and massarr[parttype]>0:   
        if (verbose):    print("filling masses according to massarr")
        if single_file:
          return np.ones(npart[parttype],dtype=dt)*massarr[parttype]
        else:
          return np.ones(nall[parttype],dtype=dt)*massarr[parttype]
  elif block=="U   ":
    data_for_type[0] = True
    block_num = 6-blocksub
  elif block=="RHO ":
    data_for_type[0] = True
    block_num = 7-blocksub
  elif block=="VOL ":
    data_for_type[0] = True
    block_num = 8-blocksub 
  elif block=="CMCE":
    data_for_type[0] = True
    dt = np.dtype((np.float32,3))
    block_num = 9-blocksub 
  elif block=="AREA":
    data_for_type[0] = True
    block_num = 10-blocksub
  elif block=="NFAC":
    data_for_type[0] = True
    dt = np.dtype(np.int64)        #depends on code version, most recent hast int32, old MyIDType   
    block_num = 11-blocksub
  elif block=="NE  ":
    data_for_type[0] = True
    block_num = 8+blockadd-blocksub
  elif block=="NH  ":
    data_for_type[0] = True
    block_num = 9+blockadd-blocksub
  elif block=="HSML":
    data_for_type[0] = True
    block_num = 10+blockadd-blocksub
  elif block=="SFR ":
    data_for_type[0] = True
    block_num = 11+blockadd-blocksub
  elif block=="MHI ":                  #This is only for the bias_HI project
    data_for_type[0] = True            #This is only for the bias_HI project
    block_num = 12+blockadd-blocksub   #This is only for the bias_HI project
  elif block=="TEMP":                  #This is only for the bias_HI project
    data_for_type[0] = True            #This is only for the bias_HI project
    block_num = 13+blockadd-blocksub   #This is only for the bias_HI project
  elif block=="AGE ":
    data_for_type[4] = True
    block_num = 12+blockadd-blocksub
  elif block=="Z   ":
    data_for_type[0] = True
    data_for_type[4] = True
    block_num = 13+blockadd-blocksub
  elif block=="BHMA":
    data_for_type[5] = True
    block_num = 14+blockadd-blocksub
  elif block=="BHMD":
    data_for_type[5] = True
    block_num = 15+blockadd-blocksub
  else:
    print("Sorry! Block type", block, "not known!")
    sys.exit()
  # - end of block description -
  
  actual_data_for_type = np.copy(data_for_type)  
  if parttype >= 0:
    actual_data_for_type[:] = False
    actual_data_for_type[parttype] = True
    if data_for_type[parttype]==False:
      print("Error: no data for specified particle type", parttype, "in the block", block)
      sys.exit()
  elif block=="MASS":
    actual_data_for_type[:] = True  
    
  allpartnum = np.int64(0)
  species_offset = np.zeros(6,np.int64)
  for j in range(6):
    species_offset[j] = allpartnum
    if actual_data_for_type[j]:
        if single_file:
            allpartnum += npart[j]
        else:
          allpartnum += nall[j]

  # define the array containing the information
  data = np.empty(allpartnum,dt)

  # loop over all subfiles    
  for i in range(filenum):

    if not(single_file) and filenum>1: 
        curfilename = filename+"."+str(i)
      
    if i>0:
      head  = snapshot_header(curfilename)
      npart = head.npart  
      del head
      
    curpartnum = np.int32(0)
    cur_species_offset = np.zeros(6,np.int64)
    for j in range(6):
      cur_species_offset[j] = curpartnum
      if data_for_type[j]:
        curpartnum += npart[j]
    
    if parttype>=0:
      actual_curpartnum = npart[parttype]      
      add_offset = cur_species_offset[parttype] 
    else:
      actual_curpartnum = curpartnum
      add_offset = np.int32(0)
      
    offset,blocksize = find_block(curfilename,format,swap,block,block_num)
    
    if i==0: # fix data type for ID if long IDs are used
      if block=="ID  ":
        if blocksize == np.dtype(dt).itemsize*curpartnum * 2:
          dt = np.uint64 
        
    if np.dtype(dt).itemsize*curpartnum != blocksize:
      print("something wrong with blocksize! expected =",np.dtype(dt).itemsize*curpartnum,"actual =",blocksize)
      sys.exit()
    
    f = open(curfilename,'rb')
    f.seek(offset + add_offset*np.dtype(dt).itemsize, os.SEEK_CUR)  
    curdat = np.fromfile(f,dtype=dt,count=actual_curpartnum) # read data
    f.close()  
    if swap:  curdat.byteswap(True)  
          
    for j in range(6):
      if actual_data_for_type[j]:
        if block=="MASS" and massarr[j]>0: # add mass block for particles for which the mass is specified in the snapshot header
          data[species_offset[j]:species_offset[j]+npart[j]] = massarr[j]
        else:
          if parttype>=0:
            data[species_offset[j]:species_offset[j]+npart[j]] = curdat
          else:
            data[species_offset[j]:species_offset[j]+npart[j]] = curdat[cur_species_offset[j]:cur_species_offset[j]+npart[j]]
        species_offset[j] += npart[j]

    del curdat

    if single_file:  break


  if physical_velocities and block=="VEL " and redshift!=0:
    data *= math.sqrt(time)

  return data
  
# ----- list all data blocks in a format 2 snapshot file -----

def list_format2_blocks(filename):
  if   os.path.exists(filename):        curfilename = filename
  elif os.path.exists(filename+".0"):   curfilename = filename+".0"
  else:
    print("file not found:", filename);  sys.exit()
  
  head   = snapshot_header(curfilename)
  format = head.format
  swap   = head.swap;  del head
  
  print('GADGET FORMAT ',format)
  if (format != 2):  print("#   OFFSET   SIZE")
  else:              print("#   BLOCK   OFFSET   SIZE")
  print("-------------------------")
  
  find_block(curfilename, format, swap, "XXXX", 0, only_list_blocks=True)
  
  print("-------------------------")

def read_gadget_header(filename):
  if   os.path.exists(filename):      curfilename = filename
  elif os.path.exists(filename+".0"): curfilename = filename+".0"
  else:
    print("file not found:", filename);  sys.exit()

  head=snapshot_header(curfilename)
  print('npar=',head.npart)
  print('nall=',head.nall)
  print('a=',head.time)
  print('z=',head.redshift)
  print('masses=',head.massarr*1e10,'Msun/h')
  print('boxsize=',head.boxsize,'kpc/h')
  print('filenum=',head.filenum)
  print('cooling=',head.cooling)
  print('Omega_m,Omega_l=',head.omega_m,head.omega_l)
  print('h=',head.hubble,'\n')
  
  rhocrit=2.77536627e11 #h**2 M_sun/Mpc**3
  rhocrit=rhocrit/1e9 #h**2M_sun/kpc**3
  
  Omega_CDM=head.nall[1]*head.massarr[1]*1e10/(head.boxsize**3*rhocrit)
  print('DM mass=%.5e  Omega_DM = %.5f'\
    %(head.massarr[1]*1e10, Omega_CDM))
  if head.nall[2]>0 and head.massarr[2]>0:
    Omega_NU=head.nall[2]*head.massarr[2]*1e10/(head.boxsize**3*rhocrit)
    print('NU mass=%.5e  Omega_NU = %.5f'\
        %(head.massarr[2]*1e10, Omega_NU))
    print('Sum of neutrino masses=%.5f eV'\
        %(Omega_NU*head.hubble**2*94.1745))
