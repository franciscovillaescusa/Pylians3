*****************************
Hydrodynamic simulations (3D)
*****************************

In hydrodynamic simulations, gas is usually modelled as spheres or voronoi cells. In this case, instead of using the standard mass assignment schemes such as NPG, CIC or TSC, it is better to associate these spheres to the regular grid. We recomment using this code to achieve this:

`voxelize <https://github.com/leanderthiele/voxelize>`_
