import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

rlt_dir = "results/pix256_framerate1_lr1e-6/latest"
f = "{}/result.npy".format(rlt_dir)
result = np.load(f)

fig, ax = plt.subplots()
ax.plot(result[0], result[1], 'ro', result[0], result[2], 'bs')

ax.set(xlabel='video Id', ylabel='score')
ax.grid()

fig.savefig("{}/result.png".format(rlt_dir))
#plt.show()

