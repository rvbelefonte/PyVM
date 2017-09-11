import numpy as np
import copy
import matplotlib.pyplot as plt

from pyvm.models.vm import VM

vm = VM('Myers_3D.vm')

rf0 = copy.copy(vm.rf[0][:, :])

fig = plt.figure(figsize=(20, 10))

ax = fig.add_subplot(131)
ax.imshow(np.flipud(rf0.T))

plt.title('Before smoothing')

vm.smooth_interface(0, nwin=25)

ax = fig.add_subplot(132)
ax.imshow(np.flipud(vm.rf[0].T))
plt.title('After smoothing')

ax = fig.add_subplot(133)
ax.imshow(np.flipud((vm.rf[0] - rf0).T), cmap='seismic')
plt.title('Difference')

plt.show()
