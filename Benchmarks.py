## Funções Benchmark
# 
# São utilizadas para testar o desempenho do algoritmo, visto que o ótimo dessas funções ja é conhecido


import numpy as np
import math

def Rastrigin(xx):

	## RASTRIGIN FUNCTION
	#
	# For function details and reference information, see:
	# http://www.sfu.ca/~ssurjano/
	#
	# INPUT:
	# xx = [x1, x2, ..., xd]
	#
	# Mínimo Global: [0, ..., 0]

	sum = 0
	for i in range(len(xx)):
		xi = xx[i]
		sum += (xi**2 - 10*np.cos(2*np.pi*xi))

	return 10*len(xx) + sum


def Griewank(xx):

	## GRIEWANK FUNCTION
	#
	# For function details and reference information, see:
	# http://www.sfu.ca/~ssurjano/
	#
	# INPUT:
	# xx = [x1, x2, ..., xd]
	#
	# Mínimo Global: [0, ..., 0]

	sum = 0
	prod = 1

	for i in range(len(xx)):
		xi = xx[i]
		sum += xi**2/4000
		prod = prod * np.cos(xi/np.sqrt(i))

	return sum - prod + 1;