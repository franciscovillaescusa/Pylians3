"""
Routines for reading headers and data blocks from Gadget and Subfind snapshot files (format 2 only)
usage e.g.:

import readsnap as rs
header = rs.snapshot_header("snap_063.0") # reads snapshot header
print header.massarr
mass = rs.read_block("snap_063","MASS") 
print mass[0:10]
To print the INFO block just execute the following:
rs.read_block("snap_063","INFO") 


for mutliple file snapshots give e.g. the filename "snap_063" rather than "snap_063.0" to read_block
the returned data block is ordered by particle species even when read from a multiple file snapshot


NS: this script is not (yet) able to read the snap that rely upon the 'massarr' array

"""
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
    self.nall = np.fromfile(f,dtype=np.int32,count=6)
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

def find_block(filename, swap, block, block_num, only_list_blocks=False):
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
    f.seek(4, os.SEEK_CUR)
    curblock = f.read(4)
    if (block == curblock):
      found = True
    f.seek(8, os.SEEK_CUR)  
        
    curblocksize = (np.fromfile(f,dtype=np.uint32,count=1))[0]
    if swap:
      curblocksize = curblocksize.byteswap()
    
    # - print some debug info about found data blocks -
    #print curblock, curblock_num, curblocksize
    only_list_blocks = False
    if only_list_blocks:
      print(curblock_num,curblock,f.tell(),curblocksize)
      found = False
        
    
    if found:
      blocksize = curblocksize
      offset = f.tell()
    else:
      f.seek(curblocksize, os.SEEK_CUR)
      blocksize_check = (np.fromfile(f,dtype=np.uint32,count=1))[0]
      if swap: blocksize_check = blocksize_check.byteswap()
      if (curblocksize != blocksize_check):
        print("something wrong",curblocksize,blocksize_check)
        sys.exit()
      curblock_num += 1
  f.close()
      
  if ((not found) and (not only_list_blocks)):
    return -1,-1
    
  if (not only_list_blocks):
    return offset,blocksize
 








# ----- read the INFO block ----- 

def get_info(curfilename,block,swap):
  # Open the file
  f = open (curfilename,'rb')  
  f.seek(0, os.SEEK_END)
  filesize = f.tell()
  f.seek(0, os.SEEK_SET)

  # Look for the 'INFO' field
  found = False
  while ((not found) and (f.tell()<filesize)):
    f.seek(4, os.SEEK_CUR)
    label = f.read(4)
    if (label == 'INFO'):
      found = True
      break
    f.seek(8, os.SEEK_CUR)  
    curblocksize = (np.fromfile(f,dtype=np.uint32,count=1))[0]
    if swap:
      curblocksize = curblocksize.byteswap()
    f.seek(curblocksize+4, os.SEEK_CUR)

  # If the INFO block is present, read it
  if (found == True):
    dummy = f.read(8)
    blcksz = np.fromfile(f,dtype=np.int32,count=1)
    if swap:
      blcksz = blcksz.byteswap()
    nfields = blcksz//(12+4*7)
    myname = 'XXXX'
    for cnt in range(nfields):
      myname = f.read(4)
      mytype = f.read(8)
      myinfo = np.fromfile(f,dtype=np.uint32,count=7)
      if swap:
        myinfo = myinfo.byteswap()
      if (block == 'INFO'):
        print(myname,':\t',mytype,' ',myinfo)
      cnt += 1
      if (myname == block):
        f.close()
        return mytype,myinfo[0],myinfo[1:]
    if (block == 'INFO'):
      return 0,0,0
    print('The block you are asking for is not present in the INFO block')
    sys.exit()

  # If there is no INFO block, then assume the format
  if (block == 'MASS'):
    f.close()
    return 'FLOAT   ',1,np.array([1,1,1,1,1,1])
  elif (block in ['POS ','VEL ']):
    f.close()
    return 'FLOATN  ',3,np.array([1,1,1,1,1,1])
  elif (block == 'ID  '):
    f.close()
    return 'LONG    ',1,np.array([1,1,1,1,1,1])
  else:
    print('In this snapshot there is no INFO field and you are asking for a block which is not one of the following: MASS, POS, VEL, ID')
    sys.exit()
  




# ----- read data block -----
def read_block(filename, block):
  
  if os.path.exists(filename):
    curfilename = filename
  elif os.path.exists(filename+".0"):
    curfilename = filename+".0"
  else:
    print("file not found:", filename)
    sys.exit()
  
  head = snapshot_header(curfilename)
  swap = head.swap
  npart = head.npart # number of particles in this file
  nall = head.nall # total number of particles
  massarr = head.massarr
  filenum = head.filenum
  redshift = head.redshift
  time = head.time
 
  if (block == 'HEAD'):
    return head # the program stops here

  del head

  # read the INFO block and get the information on the fields. 
  # If there is no INFO block, then the usual formats are assumed (POS, VEL, MASS, ID)
  (datatype,dimdata,data_for_type) = get_info(curfilename,block,swap)
  if (datatype == 0):
    return 0 # The INFO block has been printed and now the program stops

  if (datatype == 'FLOAT   '):
    dt = np.dtype(np.float32)    
  elif (datatype == 'FLOATN  '):
    dt = np.dtype((np.float32,dimdata))    
  if (datatype == 'DOUBLE   '):
    dt = np.dtype(np.float64)    
  elif (datatype == 'DOUBLEN  '):
    dt = np.dtype((np.float64,dimdata))    
  elif (datatype == 'LONG    '):
    dt = np.dtype(np.uint32)    
  elif (datatype == 'LONGN   '):
    dt = np.dtype((np.uint32,dimdata))    
  elif (datatype == 'LLONG   '):
    dt = np.dtype(np.uint64)    
  elif (datatype == 'LLONGN  '):
    dt = np.dtype((np.uint64,dimdata))    

    
  allpartnum = np.int64(0)
  species_offset = np.zeros(6,np.int64)
  for j in range(6):
    species_offset[j] = allpartnum
    if data_for_type[j]:
      allpartnum += nall[j]

  for i in range(filenum): # main loop over files
    if filenum>1:
      curfilename = filename+"."+str(i)
      
    if i>0:
      head = snapshot_header(curfilename)
      npart = head.npart  
      del head
      
    curpartnum = np.int32(0)
    cur_species_offset = np.zeros(6,np.int64)
    for j in range(6):
      cur_species_offset[j] = curpartnum
      if data_for_type[j]:
        curpartnum += npart[j]
    
    # get the offset and the lenght of the block
    offset,blocksize = find_block(curfilename,swap,block,1)
    if (offset == -1 & i == 0):
      print("Error: block not found")
      sys.exit()
    elif ((offset == -1) & (i > 0)):
      return data # the program stops here


    if i==0: # fix data type for ID if long IDs are used
      if block=="ID  ":
        if blocksize == np.dtype(dt).itemsize*curpartnum * 2:
          dt = np.uint64 
        
    if np.dtype(dt).itemsize*curpartnum != blocksize:
      print("something wrong with blocksize! expected =",np.dtype(dt).itemsize*curpartnum,"actual =",blocksize)
      print("itemsize,curpartnum = ",np.dtype(dt).itemsize,curpartnum)
      sys.exit()
    
    f = open(curfilename,'rb')
    f.seek(offset, os.SEEK_CUR)  
    curdat = np.fromfile(f,dtype=dt,count=curpartnum) # read data
    f.close()  
    if swap:
      curdat.byteswap(True)  
      
    if i==0:
      data = np.ones(allpartnum,dt)
    
    for j in range(6):
      if data_for_type[j]:
        if block=="MASS" and massarr[j]>0: # add mass block for particles for which the mass is specified in the snapshot header
          data[species_offset[j]:species_offset[j]+npart[j]] = massarr[j]
        else:
          data[species_offset[j]:species_offset[j]+npart[j]] = curdat[cur_species_offset[j]:cur_species_offset[j]+npart[j]]
        species_offset[j] += npart[j]

    del curdat

  return data
  













# ----- list all data blocks in a format 2 snapshot file -----

def list_format2_blocks(filename):
  if os.path.exists(filename):
    curfilename = filename
  elif os.path.exists(filename+".0"):
    curfilename = filename+".0"
  else:
    print("file not found:", filename)
    sys.exit()
  
  head = snapshot_header(curfilename)
  format = head.format
  swap = head.swap
  del head
  
  print('GADGET FORMAT ',format)
  if (format != 2):
    print("#   OFFSET   SIZE")
  else:            
    print("#   BLOCK   OFFSET   SIZE")
  print("-------------------------")
  
  find_block(curfilename, format, swap, "XXXX", 0, only_list_blocks=True)
  
  print("-------------------------")

def read_gadget_header(filename):
  if os.path.exists(filename):
    curfilename = filename
  elif os.path.exists(filename+".0"):
    curfilename = filename+".0"
  else:
    print("file not found:", filename)
    sys.exit()

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
  
  Omega_DM=head.nall[1]*head.massarr[1]*1e10/(head.boxsize**3*rhocrit)
  print('DM mass=',head.massarr[1]*1e10,'Omega_DM=',Omega_DM)
  if head.nall[2]>0 and head.massarr[2]>0:
    Omega_NU=head.nall[2]*head.massarr[2]*1e10/(head.boxsize**3*rhocrit)
    print('NU mass=',head.massarr[2]*1e10,'Omega_NU=',Omega_NU)
    print('Sum of neutrino masses=',Omega_NU*head.hubble**2*94.1745,'eV')
