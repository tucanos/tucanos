import numpy as np
from scipy.spatial import cKDTree
from time import time

pts0 = np.random.rand(230945, 3)

pts1 = np.random.rand(230945, 3)
t1 = time()
tree = cKDTree(pts0)
t2 = time()
print(f"build: {t2 - t1}s")

t1 = time()
tree.query(pts1)
t2 = time()
print(f"query: {t2 - t1}s")

print(pts0.shape)
