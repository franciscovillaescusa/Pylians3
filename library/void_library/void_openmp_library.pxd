cdef extern from "void_openmp_library.h":

      void mark_void_region(char *in_void, int Ncells, int dims, float R_grid2,
                            int i, int j, int k, int threads)

      int num_voids_around(long total_voids_found, int dims, 
                           float middle, int i, int j, int k, 
                           float *void_radius, int *void_pos, float R_grid, 
                           int threads)

      int num_voids_around2(int Ncells, int i, int j, int k, int dims, 
                            float R_grid2, char *in_void, int threads)
