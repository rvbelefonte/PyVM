"""
Build a simple VM model from scratch
"""
import numpy as np
from pyvm.models.vm import VM

# create a 2D model
vm = VM(shape=(512, 1, 256), spacing=(0.5, 1, 0.1),
        origin=(412, 412, -2))

# add interfaces
vm.insert_interface(0)
vm.insert_interface(3)
vm.insert_interface(5)
vm.insert_interface(12)

# add velocities
vm.define_constant_layer_velocity(0, 0.333)
vm.define_stretched_layer_velocities(1, vel=[1.49, 1.51])
vm.define_stretched_layer_velocities(2, vel=[None, 2.3])
vm.define_stretched_layer_velocities(3, vel=[4.4, 6.8, 6.9, 7.2])
vm.define_constant_layer_gradient(4, 0.1)

# plot
vm.plot(aspect=10)
