import numpy as np
import matplotlib.pyplot as plt

fig, (ax0, ax1) = plt.subplots(2, 1)
old = np.loadtxt("cybe-cylinder_old_4_q.dat")
new = np.loadtxt("cybe-cylinder_ordered_4_q.dat")
print(old.min(), old.max())
print(new.min(), new.max())

ax0.hist(old, bins=20, alpha=0.25, label="old")
ax0.hist(new, bins=20, alpha=0.25, label="new")
ax0.legend(loc="best")

old = np.loadtxt("cybe-cylinder_old_4_l.dat")
new = np.loadtxt("cybe-cylinder_ordered_4_l.dat")
print(old.min(), old.max())
print(new.min(), new.max())

ax1.hist(old, bins=20, alpha=0.25, label="old")
ax1.hist(new, bins=20, alpha=0.25, label="new")


fig.savefig("fig.png")
