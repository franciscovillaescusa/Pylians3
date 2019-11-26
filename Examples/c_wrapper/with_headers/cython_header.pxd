cdef extern from "ccode.h":
	void cprint()
	double csum(double *a, long elements)
	double cfunc(double a)
