***************************
Density field interpolation
***************************

Sometimes we have a density field and we would like to evalute it at some positions (e.g. particle positions, void positions...etc). Pylians provides the routine ``CIC_interp`` to accomplish this. That routine uses the Cloud-in-Cell (CIC) scheme to interpolate the density field into the particle positions. The ingredients are:

- ``field``. This is the density field. Should be a 3D float numpy array
- ``BoxSize``. Size of the density field. Units in Mpc/h
- ``pos``. These are the positions of the particles. Units in Mpc/h
- ``density_interpolated``. This is a 1D array with the value of the density field at the particle positions. Should be a numpy float array.

An example is this:
  
.. code-block:: python
		
   import MAS_library as MASL

   # define the array containing the value of the density field at positions pos
   density_interpolated = np.zeros(pos.shape[0], dtype=np.float32)

   # find the value of the density field at the positions pos
   MASL.CIC_interp(field, BoxSize, pos, density_interpolated)
