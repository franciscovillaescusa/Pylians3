#include <stdio.h>
#include "void_openmp_library.h"
#include <omp.h>
#include <math.h>


void mark_void_region(char *in_void, int Ncells, int dims, float R_grid2,
		      int i, int j, int k, int threads)
{
  int l, m, n, i1, j1, k1;
  long number;
  float dist2;

#pragma omp parallel for num_threads(threads) private(l,m,n,i1,j1,k1,dist2,number) firstprivate(i,j,k,Ncells,R_grid2,dims) shared(in_void)
  for (l=-Ncells; l<=Ncells; l++)
    {
      //i1 = (i+l+dims)%dims;
      i1 = i+l;
      if (i1>=dims) i1 = i1-dims;
      if (i1<0)     i1 = i1+dims;
		      
      for (m=-Ncells; m<=Ncells; m++)
	{
	  //j1 = (j+m+dims)%dims;
	  j1 = j+m;
	  if (j1>=dims) j1 = j1-dims;
	  if (j1<0)     j1 = j1+dims;

	  for (n=-Ncells; n<=Ncells; n++)
	    {
	      //k1 = (k+n+dims)%dims;
	      k1 = k+n;
	      if (k1>=dims) k1 = k1-dims;
	      if (k1<0)     k1 = k1+dims;

	      dist2 = l*l + m*m + n*n;
	      if (dist2<R_grid2)
		{
		  number = dims*(i1*dims + j1) + k1;
		  in_void[number] = 1;
		}
	    }
	}
    } 
}

// This routine computes the distance between a cell and voids already identified
// if that distance is smaller than the sum of their radii then the cell can not
// host a void as it will overlap with the other void
int num_voids_around(long total_voids_found, int dims, float middle,
		     int i, int j, int k, float *void_radius, int *void_pos,
		     float R_grid, int threads)
{

  int l, nearby_voids=0;
  int dx, dy, dz, dist2;

#pragma omp parallel for num_threads(threads) private(l,dx,dy,dz,dist2)
  for (l=0; l<total_voids_found; l++)
    {
      if (nearby_voids>0)  continue;

      dx = i - void_pos[3*l+0];
      if (dx>middle)   dx = dx - dims;
      if (dx<-middle)  dx = dx + dims;

      dy = j - void_pos[3*l+1];
      if (dy>middle)   dy = dy - dims;
      if (dy<-middle)  dy = dy + dims;

      dz = k - void_pos[3*l+2];
      if (dz>middle)   dz = dz - dims;
      if (dz<-middle)  dz = dz + dims;

      dist2 = dx*dx + dy*dy + dz*dz;

      if (dist2<((void_radius[l]+R_grid)*(void_radius[l]+R_grid)))
	{
#pragma omp atomic
	  nearby_voids += 1;
	}
    }
  
  return nearby_voids;
}


// This routine looks at the cells around a given cell to see if those belong
// to other voids
int num_voids_around2(int Ncells, int i, int j, int k, int dims,
		      float R_grid2, char *in_void, int threads)
{
  int l, m, n, i1, j1, k1, nearby_voids=0;
  long num;
  float dist2;

#pragma omp parallel for num_threads(threads) private(l, m, n, i1, j1, k1, num, dist2)
  for (l=-Ncells; l<=Ncells; l++)
    {
      if (nearby_voids>0)  continue;

      //i1 = (i+l+dims)%dims;
      i1 = i+l;
      if (i1>=dims) i1 = i1-dims;
      if (i1<0)     i1 = i1+dims;

      for (m=-Ncells; m<=Ncells; m++)
	{

	  //j1 = (j+m+dims)%dims;
	  j1 = j+m;
	  if (j1>=dims) j1 = j1-dims;
	  if (j1<0)     j1 = j1+dims;

	  for (n=-Ncells; n<=Ncells; n++)
	    {

	      //k1 = (k+n+dims)%dims;
	      k1 = k+n;
	      if (k1>=dims) k1 = k1-dims;
	      if (k1<0)     k1 = k1+dims;

	      num = dims*(i1*dims + j1) + k1;

	      if (in_void[num]==0)  continue;
	      else 
		{
		  dist2 = l*l + m*m + n*n;
		  if (dist2<R_grid2)
		    {
#pragma omp atomic
		      nearby_voids += 1;
		    }
		}
	    }
	}
    }
	  
  return nearby_voids;
}

