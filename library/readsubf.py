# code for reading Subfind's subhalo_tab files
# usage e.g.:
#
# import readsubf
# cat = readsubf.subfind_catalog("./m_10002_h_94_501_z3_csf/",63,masstab=True)
# print cat.nsubs
# print "largest halo x position = ",cat.sub_pos[0][0] 

import numpy as np
import os
import sys
 
class subfind_catalog:
  def __init__(self, basedir, snapnum, group_veldisp = False, masstab = False, long_ids = False, swap = False):
    self.filebase = basedir + "/groups_" + str(snapnum).zfill(3) + "/subhalo_tab_" + str(snapnum).zfill(3) + "."
 
    #print
    #print "reading subfind catalog for snapshot",snapnum,"of",basedir
 
    if long_ids: self.id_type = np.uint64
    else: self.id_type = np.uint32
 
    self.group_veldisp = group_veldisp
    self.masstab = masstab
 
    filenum = 0
    doneflag = False
    skip_gr = 0
    skip_sub = 0
    while not doneflag:
      curfile = self.filebase + str(filenum)
      
      if (not os.path.exists(curfile)):
        print("file not found:", curfile)
        sys.exit()
      
      f = open(curfile,'rb')
              
      ngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
      totngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
      nids = np.fromfile(f, dtype=np.uint32, count=1)[0]
      totnids = np.fromfile(f, dtype=np.uint64, count=1)[0]
      ntask = np.fromfile(f, dtype=np.uint32, count=1)[0]
      nsubs = np.fromfile(f, dtype=np.uint32, count=1)[0]
      totnsubs = np.fromfile(f, dtype=np.uint32, count=1)[0]
      
      if swap:
        ngroups = ngroups.byteswap()
        totngroups = totngroups.byteswap()
        nids = nids.byteswap()
        totnids = totnids.byteswap()
        ntask = ntask.byteswap()
        nsubs = nsubs.byteswap()
        totnsubs = totnsubs.byteswap()
      
      if filenum == 0:
        self.ngroups = totngroups
        self.nids = totnids
        self.nfiles = ntask
        self.nsubs = totnsubs

        self.group_len = np.empty(totngroups, dtype=np.uint32)
        self.group_offset = np.empty(totngroups, dtype=np.uint32)
        self.group_mass = np.empty(totngroups, dtype=np.float32)
        self.group_pos = np.empty(totngroups, dtype=np.dtype((np.float32,3)))
        self.group_m_mean200 = np.empty(totngroups, dtype=np.float32)
        self.group_r_mean200 = np.empty(totngroups, dtype=np.float32)
        self.group_m_crit200 = np.empty(totngroups, dtype=np.float32)
        self.group_r_crit200 = np.empty(totngroups, dtype=np.float32)
        self.group_m_tophat200 = np.empty(totngroups, dtype=np.float32)
        self.group_r_tophat200 = np.empty(totngroups, dtype=np.float32)
        if group_veldisp:
          self.group_veldisp_mean200 = np.empty(totngroups, dtype=np.float32)
          self.group_veldisp_crit200 = np.empty(totngroups, dtype=np.float32)
          self.group_veldisp_tophat200 = np.empty(totngroups, dtype=np.float32)
        self.group_contamination_count = np.empty(totngroups, dtype=np.uint32)
        self.group_contamination_mass = np.empty(totngroups, dtype=np.float32)
        self.group_nsubs = np.empty(totngroups, dtype=np.uint32)
        self.group_firstsub = np.empty(totngroups, dtype=np.uint32)
        
        self.sub_len = np.empty(totnsubs, dtype=np.uint32)
        self.sub_offset = np.empty(totnsubs, dtype=np.uint32)
        self.sub_parent = np.empty(totnsubs, dtype=np.uint32)
        self.sub_mass = np.empty(totnsubs, dtype=np.float32)
        self.sub_pos = np.empty(totnsubs, dtype=np.dtype((np.float32,3)))
        self.sub_vel = np.empty(totnsubs, dtype=np.dtype((np.float32,3)))
        self.sub_cm = np.empty(totnsubs, dtype=np.dtype((np.float32,3)))
        self.sub_spin = np.empty(totnsubs, dtype=np.dtype((np.float32,3)))
        self.sub_veldisp = np.empty(totnsubs, dtype=np.float32)
        self.sub_vmax = np.empty(totnsubs, dtype=np.float32)
        self.sub_vmaxrad = np.empty(totnsubs, dtype=np.float32)
        self.sub_halfmassrad = np.empty(totnsubs, dtype=np.float32)
        self.sub_id_mostbound = np.empty(totnsubs, dtype=self.id_type)
        self.sub_grnr = np.empty(totnsubs, dtype=np.uint32)
        if masstab:
          self.sub_masstab = np.empty(totnsubs, dtype=np.dtype((np.float32,6)))
     
      if ngroups > 0:
        locs = slice(skip_gr, skip_gr + ngroups)
        self.group_len[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)
        self.group_offset[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)
        self.group_mass[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        self.group_pos[locs] = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=ngroups)
        self.group_m_mean200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        self.group_r_mean200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        self.group_m_crit200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        self.group_r_crit200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        self.group_m_tophat200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        self.group_r_tophat200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        if group_veldisp:
          self.group_veldisp_mean200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
          self.group_veldisp_crit200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
          self.group_veldisp_tophat200[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        self.group_contamination_count[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)
        self.group_contamination_mass[locs] = np.fromfile(f, dtype=np.float32, count=ngroups)
        self.group_nsubs[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)
        self.group_firstsub[locs] = np.fromfile(f, dtype=np.uint32, count=ngroups)        
        skip_gr += ngroups
        
      if nsubs > 0:
        locs = slice(skip_sub, skip_sub + nsubs)
        self.sub_len[locs] = np.fromfile(f, dtype=np.uint32, count=nsubs)
        self.sub_offset[locs] = np.fromfile(f, dtype=np.uint32, count=nsubs)
        self.sub_parent[locs] = np.fromfile(f, dtype=np.uint32, count=nsubs)
        self.sub_mass[locs] = np.fromfile(f, dtype=np.float32, count=nsubs)
        self.sub_pos[locs] = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=nsubs)
        self.sub_vel[locs] = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=nsubs)
        self.sub_cm[locs] = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=nsubs)
        self.sub_spin[locs] = np.fromfile(f, dtype=np.dtype((np.float32,3)), count=nsubs)
        self.sub_veldisp[locs] = np.fromfile(f, dtype=np.float32, count=nsubs)
        self.sub_vmax[locs] = np.fromfile(f, dtype=np.float32, count=nsubs)
        self.sub_vmaxrad[locs] = np.fromfile(f, dtype=np.float32, count=nsubs)
        self.sub_halfmassrad[locs] = np.fromfile(f, dtype=np.float32, count=nsubs)
        self.sub_id_mostbound[locs] = np.fromfile(f, dtype=self.id_type, count=nsubs)
        self.sub_grnr[locs] = np.fromfile(f, dtype=np.uint32, count=nsubs)
        if masstab:
          self.sub_masstab[locs] = np.fromfile(f, dtype=np.dtype((np.float32,6)), count=nsubs)
        skip_sub += nsubs

      curpos = f.tell()
      f.seek(0,os.SEEK_END)
      if curpos != f.tell(): print("Warning: finished reading before EOF for file",filenum)
      f.close()  
      #print 'finished with file number',filenum,"of",ntask
      filenum += 1
      if filenum == self.nfiles: doneflag = True
       
    if swap:
      self.group_len.byteswap(True)
      self.group_offset.byteswap(True)
      self.group_mass.byteswap(True)
      self.group_pos.byteswap(True)
      self.group_m_mean200.byteswap(True)
      self.group_r_mean200.byteswap(True)
      self.group_m_crit200.byteswap(True)
      self.group_r_crit200.byteswap(True)
      self.group_m_tophat200.byteswap(True)
      self.group_r_tophat200.byteswap(True)
      if group_veldisp:
        self.group_veldisp_mean200.byteswap(True)
        self.group_veldisp_crit200.byteswap(True)
        self.group_veldisp_tophat200.byteswap(True)
      self.group_contamination_count.byteswap(True)
      self.group_contamination_mass.byteswap(True)
      self.group_nsubs.byteswap(True)
      self.group_firstsub.byteswap(True)
        
      self.sub_len.byteswap(True)
      self.sub_offset.byteswap(True)
      self.sub_parent.byteswap(True)
      self.sub_mass.byteswap(True)
      self.sub_pos.byteswap(True)
      self.sub_vel.byteswap(True)
      self.sub_cm.byteswap(True)
      self.sub_spin.byteswap(True)
      self.sub_veldisp.byteswap(True)
      self.sub_vmax.byteswap(True)
      self.sub_vmaxrad.byteswap(True)
      self.sub_halfmassrad.byteswap(True)
      self.sub_id_mostbound.byteswap(True)
      self.sub_grnr.byteswap(True)
      if masstab:
        self.sub_masstab.byteswap(True)
       
    #print
    #print "number of groups =", self.ngroups
    #print "number of subgroups =", self.nsubs
    #if self.nsubs > 0:
    #  print "largest group of length",self.group_len[0],"has",self.group_nsubs[0],"subhalos"
    #  print



# code for reading Subfind's ID files
# usage e.g.:
#
# import readsubf
# ids = readsubf.subf_ids("./m_10002_h_94_501_z3_csf/", 0, 100)


class subf_ids:
  def __init__(self, basedir, snapnum, substart, sublen, swap = False, verbose = False, long_ids = False, read_all = False):
    self.filebase = basedir + "/groups_" + str(snapnum).zfill(3) + "/subhalo_ids_" + str(snapnum).zfill(3) + "."

    if (verbose):
      print("reading subhalo IDs for snapshot",snapnum,"of",basedir)
 
    if long_ids: self.id_type = np.uint64
    else: self.id_type = np.uint32

 
    filenum = 0
    doneflag = False
    count=substart
    found=0


    while not doneflag:
      curfile = self.filebase + str(filenum)
      
      if (not os.path.exists(curfile)):
        print("file not found:", curfile)
        sys.exit()
      
      f = open(curfile,'rb')
              
      Ngroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
      TotNgroups = np.fromfile(f, dtype=np.uint32, count=1)[0]
      NIds = np.fromfile(f, dtype=np.uint32, count=1)[0]
      TotNids = np.fromfile(f, dtype=np.uint64, count=1)[0]
      NTask = np.fromfile(f, dtype=np.uint32, count=1)[0]
      Offset = np.fromfile(f, dtype=np.uint32, count=1)[0]


      if read_all:
              substart=0
              sublen=TotNids
      if swap:
          Ngroups = Ngroups.byteswap()
          TotNgroups = TotNgroups.byteswap()
          NIds = NIds.byteswap()
          TotNids = TotNids.byteswap()
          NTask = NTask.byteswap()
          Offset = Offset.byteswap()
      if filenum == 0:
        if (verbose):
          print("Ngroups    = ", Ngroups)
          print("TotNgroups = ", Ngroups)
          print("NIds       = ", NIds)
          print("TotNids    = ", TotNids)
          print("NTask      = ", NTask)
          print("Offset     = ", Offset)
        self.nfiles = NTask
        self.SubLen=sublen
        self.SubIDs = np.empty(sublen, dtype=self.id_type)


      if count <= Offset+NIds:
        nskip = count - Offset
        nrem = Offset + NIds - count
        if sublen > nrem:
          n_to_read = nrem
        else:
          n_to_read = sublen
        if n_to_read > 0:
          if (verbose):
            print(filenum, n_to_read)
          if nskip > 0:
            dummy=np.fromfile(f, dtype=self.id_type, count=nskip)
            if (verbose):
              print(dummy)
          locs = slice(found, found + n_to_read)
          dummy2 = np.fromfile(f, dtype=self.id_type, count=n_to_read)
          if (verbose):
            print(dummy2)
          self.SubIDs[locs]=dummy2
          found += n_to_read
        count += n_to_read
        sublen -= n_to_read

      f.close()
      filenum += 1
      if filenum == self.nfiles: doneflag = True
       
    if swap:
      self.SubIDs.byteswap(True)

 
