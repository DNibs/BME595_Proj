# BME595 Project
# Author: David Niblick
# Date: 05DEC18
# test.py


import cupy as cp
import numpy as np
import time


print(cp.cuda.Device().mem_info)

test_np1 = np.random.rand(1000, 1000)
test_np2 = np.random.rand(1000, 1000)

test_cp1 = cp.random.rand(1000, 1000)
test_cp2 = cp.random.rand(1000, 1000)

t0 = time.time()
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
np.matmul(test_np1, test_np2)
t1 = time.time()
print(t1-t0)

t2 = time.time()
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
cp.matmul(test_cp1, test_cp2)
t3 = time.time()
print(t3-t2)

