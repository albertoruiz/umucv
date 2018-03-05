#include "cfuns.h"

int nms(const double *m, int cols, int rows,
        const unsigned char *a,
        double * r) {

    int i,j;
    double v;
    for (i=1; i<rows-1; i++) {
        for (j=0; j<cols-1; j++) {
            v = m[i*cols+j];
            switch(a[i*cols+j]) {
                case 0: { if(v > m[(i)  *cols+j-1] && v > m[(i)  *cols+j+1]) r[i*cols+j] = v; break; }
                case 1: { if(v > m[(i-1)*cols+j-1] && v > m[(i+1)*cols+j+1]) r[i*cols+j] = v; break; }
                case 2: { if(v > m[(i-1)*cols+j]   && v > m[(i+1)*cols+j])   r[i*cols+j] = v; break; }
                case 3: { if(v > m[(i+1)*cols+j-1] && v > m[(i-1)*cols+j+1]) r[i*cols+j] = v; break; }
            }
        }
    }
    return 0;
}

