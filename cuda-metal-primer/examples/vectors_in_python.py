import numpy as np, psutil, os
N = 10_000_000

a = np.random.rand(N).astype(np.float32)
b = np.random.rand(N).astype(np.float32)

c = a + b                                 # Vector add

print("Theoretical bytes: ", 3*N*a.itemsize)
print("NumPy actually holds: ", a.nbytes+b.nbytes+c.nbytes)
print("Process RSS: ", psutil.Process(os.getpid()).memory_info().rss)

np.testing.assert_allclose(c, a+b)