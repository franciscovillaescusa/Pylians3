#include <stdio.h>
#include "MAS_c.h"
#include <omp.h>
#include <math.h>

// ###################### CIC #################### //
// This function carries out the standard CIC with weights in 3D
void CIC(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims, int axes,
	 FLOAT BoxSize, int threads)
{

  long i;
  int axis, index[3][2], a, b, c, n_max, weights, l, m, n;
  FLOAT inv_cell_size, dist, WW;
  FLOAT C[3][2]={{1,1},
		 {1,1},
		 {1,1}};

  inv_cell_size = dims*1.0/BoxSize; 

  if (axes==3) // for 3D
    {
      a = dims*dims;
      b = dims;
      c = 1;
      n_max = 2;
    }
  else //for 2D
    {
      a = dims;
      b = 1;
      c = 0;
      n_max = 1;
    }

  if (W==NULL)
    weights=0;
  else
    weights=1;
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,l,m,n,index,WW) firstprivate(a,b,c,n_max,axes,C,weights) shared(number,pos,W)
  for (i=0; i<particles; i++)
    {

      if (weights==0)
	WW = 1.0;
      else
	WW = W[i];
      
      for (axis=0; axis<axes; axis++)
	{
	  dist           = pos[axes*i+axis]*inv_cell_size;
	  C[axis][1]     = dist - (int)dist;
	  C[axis][0]     = 1.0 - C[axis][1];
	  index[axis][0] = ((int)dist)%dims;         
	  index[axis][1] = (index[axis][0] + 1)%dims;
	  //index is always be positive, no need to add dims
	}
      
      for (l=0; l<2; l++)
	for (m=0; m<2; m++)
	  for (n=0; n<n_max; n++)
	    {
#pragma omp atomic
	      number[index[0][l]*a + index[1][m]*b + index[2][n]*c] += C[0][l]*C[1][m]*C[2][n]*WW;
	    }    
    }
  
}


// ###################### NGP #################### //
// This function carries out the standard NGP w/ or w/o weights in 2D or 3D
void NGP(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims, int axes,
	 FLOAT BoxSize, int threads)
{

  long i;
  int a, b, c, axis, weights, index[3] = {0,0,0};
  FLOAT inv_cell_size, WW;

  inv_cell_size = dims*1.0/BoxSize; 

  if (axes==3) // for 3D
    {
      a = dims*dims;
      b = dims;
      c = 1;
    }
  else //for 2D
    {
      a = dims;
      b = 1;
      c = 0;
    }

  if (W==NULL)
    weights=0;
  else
    weights=1;
  
#pragma omp parallel for num_threads(threads) private(i,axis,WW) firstprivate(a,b,c,weights,index) shared(number,pos,W)
  for (i=0; i<particles; i++)
    {

      if (weights==0)
	WW = 1.0;
      else
	WW = W[i];
      
      for (axis=0; axis<axes; axis++)
	{
	  index[axis] = (int)(pos[axes*i+axis]*inv_cell_size + 0.5);
	  index[axis] = (index[axis])%dims; //Always positive. No need to add +dims
	}
#pragma omp atomic
      number[index[0]*a + index[1]*b + index[2]*c] += WW;
      
    }
}


// ###################### TSC #################### //
// This function carries out the standard TSC w/ or w/o weights in 2D or 3D
void TSC(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims, int axes,
	 FLOAT BoxSize, int threads)
{

  long i;
  int j, l, m, n, axis, index[3][3], minimum, weights;
  int a,b,c,n_max;
  FLOAT inv_cell_size, dist, diff, WW;
  FLOAT C[3][3] = {{1,1,1},
		   {1,1,1},
		   {1,1,1}};

  inv_cell_size = dims*1.0/BoxSize; 

  if (axes==3) // for 3D
    {
      a = dims*dims;
      b = dims;
      c = 1;
      n_max = 3;
    }
  else //for 2D
    {
      a = dims;
      b = 1;
      c = 0;
      n_max = 1;
    }

  if (W==NULL)
    weights=0;
  else
    weights=1;
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,minimum,j,index,diff,l,m,n,WW) firstprivate(a,b,c,n_max,axes,C,weights) shared(number,pos,W)
  for (i=0; i<particles; i++)
    {
      
      if (weights==0)
	WW = 1.0;
      else
	WW = W[i];
      
      for (axis=0; axis<axes; axis++)
	{
	  dist    = pos[axes*i+axis]*inv_cell_size;
	  minimum = (int)(floor(dist-1.5));
	  
	  for (j=0; j<3; j++)
	    {
	      index[axis][j] = (minimum+j+1+dims)%dims;
	      diff = fabs(minimum + j+1 - dist);
	      if (diff<0.5)
		C[axis][j] = 0.75-diff*diff;
	      else if (diff<1.5)
		C[axis][j] = 0.5*(1.5-diff)*(1.5-diff);
	      else
		C[axis][j] = 0.0;
	    }
	}
      for (l=0; l<3; l++)
	for (m=0; m<3; m++)
	  for (n=0; n<n_max; n++)
	    {
#pragma omp atomic
	      number[index[0][l]*a + index[1][m]*b + index[2][n]*c] += C[0][l]*C[1][m]*C[2][n]*WW;
	    }
    }
} 


// ###################### PCS #################### //
// This function carries out the standard PCS w/ or w/o weights in 2D or 3D
void PCS(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims, int axes,
	  FLOAT BoxSize, int threads)
{

  long i;
  int j, l, m, n, axis, minimum, index[3][4], weights;
  int a,b,c,n_max;
  FLOAT inv_cell_size, dist, diff, WW;
  FLOAT C[3][4] = {{1,1,1,1},
		   {1,1,1,1},
		   {1,1,1,1}};
  
  inv_cell_size = dims*1.0/BoxSize;
  
  if (axes==3) // for 3D
    {
      a = dims*dims;
      b = dims;
      c = 1;
      n_max = 4;
    }
  else //for 2D
    {
      a = dims;
      b = 1;
      c = 0;
      n_max = 1;
    }

  if (W==NULL)
    weights=0;
  else
    weights=1;
  
#pragma omp parallel for num_threads(threads) private(i,axis,dist,minimum,j,index,diff,l,m,n,WW) firstprivate(a,b,c,n_max,axes,C,weights) shared(number,pos,W)
  for (i=0; i<particles; i++)
    {

      if (weights==0)
	WW = 1.0;
      else
	WW = W[i];
      
      for (axis=0; axis<axes; axis++)
	{
	  dist    = pos[axes*i+axis]*inv_cell_size;
	  minimum = (int)(floor(dist-2.0));
	  
	  for (j=0; j<4; j++)
	    {
	      index[axis][j] = (minimum+j+1+dims)%dims;
	      diff = fabs(minimum + j+1 - dist);
	      if (diff<1.0)
		C[axis][j] = (4.0 - 6.0*diff*diff + 3.0*diff*diff*diff)/6.0;
	      else if (diff<2.0)
		C[axis][j] = (2.0 - diff)*(2.0 - diff)*(2.0 - diff)/6.0;
	      else
		C[axis][j] = 0.0; 
	    }
	}
      for (l=0; l<4; l++)
	for (m=0; m<4; m++)
	  for (n=0; n<n_max; n++)
	    {
#pragma omp atomic
	      number[index[0][l]*a + index[1][m]*b + index[2][n]*c] += C[0][l]*C[1][m]*C[2][n]*WW;
	    }
    }
}







  


