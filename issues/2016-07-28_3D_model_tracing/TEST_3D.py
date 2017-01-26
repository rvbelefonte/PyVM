import numpy as np
from pyvm.models.vm import VM

# define the model domain in terms of grid dimensions, spacing, and origin
ny = 460
vm = VM(shape=(500, ny, 100), spacing=(1, 1, 1), origin=(0, 0, -5))

##

# sloping boundary - SEA FLOOR & BATHYMETRY
specs = [#xstart, xend, slope
    [0., 230., 0],
    [230., 275., -0.05],
    [275., 330., 0],
    [330., 350., 0.07],
    [350., 400., 0],
    [400., 450., -0.04],
    [450., 500., 0]]
z0 = 0  # intial depth at left-hand side of model

# build full boundary
z = np.ones(vm.nx)
for x0, x1, m in specs:
    ix = vm.xrange2i(x0, x1)
    x = vm.grid.x[ix]
    z[ix] = z0 + m * (x - x[0])
    z0 = z[ix[-1]]
    
# expand into 3D
n = np.ones((vm.nx,1))
for i in range(0, ny):
    n[i] = z[i]

s = np.ones((vm.nx,vm.ny))
for i in range(0, ny):
    #s = np.hstack((n,n))
    s[i,:] = n[i]
    
z = np.array(s)

# add the boundary
vm.insert_interface(np.reshape(z, (vm.nx, vm.ny)))

##

# add some velocities
vm.define_constant_layer_velocity(0, 4.0) #surface
vm.define_constant_layer_velocity(1, 6.0) #ocean


print(vm)
# plot
vm.plot()

# write
vm.write('11.11.vm')
