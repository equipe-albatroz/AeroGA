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


input = [0,	1,	0.161019529607381,	1.2502671441041,	0.0276957308348257,	0.0854599174105143]  # 18.8682334373153
input1 = [3,	2,	0.0406522343922652,	0.882561171949484,	0.116428414340727,	0.958918593142206]  # 20.5290242162952
input2 = [4,	0,	0.36168562873824,	0.0318010130818993,	0,	0.625713149448781]  # 50.2171043329908
input3 = [1,	1,	0.230587361314719,	0.683895223855159,	0.218278893208412,	0.666777309528867]  # 48.8455516407746
input4 = [5,	4,	0.4,	1.5,	0.30987253443182,	0.702036985975489]  # 108.731041451515

input5 = [2,0,0.0,0.0,0.0,1.0]  # 0.0
input6 = [0,4,0.0,1.0304723084294227,0.0,0.0]  # 0.0

print(Rastrigin(input))
print(Rastrigin(input1))
print(Rastrigin(input2))
print(Rastrigin(input3))
print(Rastrigin(input4))
print(Rastrigin(input5))
print(Rastrigin(input6))