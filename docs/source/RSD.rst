.. _RSD:

**************************
Redshift-space distortions
**************************

Pylians provides the routine ``pos_redshift_space`` to displace particle positions from real-space to redshift-space. The arguments of that function are:

- ``pos``. This is an array with the co-moving positions of the particles. Should be float numpy array. Notice that this array will be overwritten with the positions of the particles in redshift-space. So if you want to keep the positions of the original particles, is better to pass a copy of this array: e.g. ``pos_RSD = np.copy(pos)``. Units should be Mpc/h.
- ``vel``. This is an array with the peculiar velocities of the particles. Should be a float numpy array. Units should be km/s
- ``BoxSize``. Size of the simulation box. Units should be Mpc/h
- ``Hubble``. Value of the Hubble constant at redshift ``redshift``. Units should be (km/s)/(Mpc/h).
- ``redshift``. The considered redshift.
- ``axis``. Redshift-space distortions are going to be place along the x-(axis=0), y-(axis=1) or z-(axis=2) axis.

.. code-block:: python
		
   import redshift_space_library as RSL

   # move particles to redshift-space. After this call, pos will contain the
   # positions of the particles in redshift-space
   RSL.pos_redshift_space(pos, vel, BoxSize, Hubble, redshift, axis)

