******
Gadget
******

Pylians provides a few routines to work with Gadget snapshots.

Snapshots
---------

The routine library ``readgadget`` can be used to read generic Gadget snapshots, with format I, II or hdf5. An example is this:


.. code-block:: python
		
    import readgadget

    # input files
    snapshot = '/home/fvillaescusa/Quijote/Snapshots/h_p/snapdir_002/snap_002'
    ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)

    # read header
    header   = readgadget.header(snapshot)
    BoxSize  = header.boxsize/1e3  #Mpc/h
    Nall     = header.nall         #Total number of particles
    Masses   = header.massarr*1e10 #Masses of the particles in Msun/h
    Omega_m  = header.omega_m      #value of Omega_m
    Omega_l  = header.omega_l      #value of Omega_l
    h        = header.hubble       #value of h
    redshift = header.redshift     #redshift of the snapshot
    Hubble   = 100.0*np.sqrt(Omega_m*(1.0+redshift)**3+Omega_l)#Value of H(z) in km/s/(Mpc/h)
    
    # read positions, velocities and IDs of the particles
    pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
    vel = readgadget.read_block(snapshot, "VEL ", ptype)     #peculiar velocities in km/s
    ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0

.. note::

   While ``readgadget`` can read generic N-nody outputs from Gadget, it may only be able to read a few blocks from hydrodynamic simulations. In this case is better to modify the library to read the particular fields that may be unique to your simulation.

The scheme below shows the traditional structure of the Format I and Format II Gadget snapshots.
   
.. image:: Format_gadget.pdf

   
Halo catalogues
---------------

The library ``readfof`` can be used to read Friends-of-Friends (FoF) halo catalogues that are written in Gadget format. An example is this:

.. code-block:: python
		
    import readfof 

    # input files
    snapdir = '/home/fvillaescusa/Quijote/Halos/s8_p/145/' #folder hosting the catalogue
    snapnum = 4                                            #redshift 0

    # determine the redshift of the catalogue
    z_dict = {4:0.0, 3:0.5, 2:1.0, 1:2.0, 0:3.0}
    redshift = z_dict[snapnum]

    # read the halo catalogue
    FoF = readfof.FoF_catalog(snapdir, snapnum, long_ids=False,
		              swap=False, SFR=False, read_IDs=False)
										
    # get the properties of the halos
    pos_h = FoF.GroupPos/1e3            #Halo positions in Mpc/h
    mass  = FoF.GroupMass*1e10          #Halo masses in Msun/h
    vel_h = FoF.GroupVel*(1.0+redshift) #Halo peculiar velocities in km/s
    Npart = FoF.GroupLen                #Number of CDM particles in the halo
