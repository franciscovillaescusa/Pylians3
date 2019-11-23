typedef float FLOAT;

void NGP(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	 int axes, FLOAT BoxSize, int threads);
void CIC(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	 int axes, FLOAT BoxSize, int threads);
void TSC(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	 int axes, FLOAT BoxSize, int threads);
void PCS(FLOAT *pos, FLOAT *number, FLOAT *W, long particles, int dims,
	 int axes, FLOAT BoxSize, int threads);
