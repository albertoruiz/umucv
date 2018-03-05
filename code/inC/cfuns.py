from ctypes import *
from numpy.ctypeslib import load_library, ndpointer
import numpy as np

# Cargar librería dinámica compilada en C,
# situada en un directorio con el mismo nombre
libcfuns = load_library('libcfuns.so', __file__[:-3])

# Tipos de la función en C:
libcfuns.nms.argtypes = [
    ndpointer(dtype=np.float64, ndim=2, flags='C'), c_int, c_int,
    ndpointer(dtype=np.uint8,   ndim=2, flags='C'),
    ndpointer(dtype=np.float64, ndim=2, flags='C')]

# Wrapper Python para llamar a la función en C
def nms(m,a):
    rows, cols = m.shape
    r = np.zeros_like(m)
    libcfuns.nms(m,cols,rows,a,r)
    return r

