"""
Test suite for the vm module
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import shutil
import doctest
import unittest
import numpy as np
from pyvm.utils.loaders import get_example_file
from pyvm.models import vm
from pyvm.models.vm import VM

class vmTestCase(unittest.TestCase):

    def test_init_model(self):
        """
        Should initialize a new model
        """
        shape = (128, 1, 64)
        vm = VM(shape=shape)

        # should have a created the 3D grid manager
        self.assertTrue(hasattr(vm, 'grid'))
        self.assertEqual(vm.grid.nx, shape[0])
        self.assertEqual(vm.grid.ny, shape[1])
        self.assertEqual(vm.grid.nz, shape[2])

        # should have default attributes needed by other routines
        for attr in ['sl', 'nr', 'nx', 'ny', 'nz', 'dx', 'dy', 'dz',
                'rf', 'jp', 'ij', 'ir', 'r1', 'r2', 'ilyr']:
            self.assertTrue(hasattr(vm, attr))

        # grid should be initialized to zeros
        self.assertEqual(vm.sl[0, 0, 0], 0)

        # setting certain aliases should set parent
        vm.sl[0, 0, 0] = 999
        self.assertEqual(vm.grid.values[0, 0, 0], 999)
        
        for attr in ['dx', 'dy', 'dz']:
            vm.__setattr__(attr, 999)
            self.assertEqual(vm.__getattribute__(attr), 999)
            self.assertEqual(vm.grid.__getattribute__(attr), 999)
                    
        vm.r1 = (9.99, 8.88, 7.77)
        self.assertEqual(vm.r1, (9.99, 8.88, 7.77))
        self.assertEqual(vm.grid.origin, (9.99, 8.88, 7.77))
        
        # setting r2 should update grid spacing
        nx0, ny0, nz0 = vm.grid.shape[:]
        dx0, dy0, dz0 = vm.grid.spacing[:]

        vm.r2 = (400, 200, 300)
        self.assertEqual(vm.nx, nx0)
        self.assertEqual(vm.ny, ny0)
        self.assertEqual(vm.nz, nz0)
        self.assertNotEqual(vm.dx, dx0)
        self.assertNotEqual(vm.dy, dy0)
        self.assertNotEqual(vm.dz, dz0)
            
        # extents must be 3D
        for attr in ['r1', 'r2']:
            with self.assertRaises(ValueError) as context:
                vm.__setattr__(attr, (0, 0))

        # should not allow certain aliases to be set
        for attr in ['nx', 'ny', 'nz', 'nr']:
            with self.assertRaises(AttributeError) as context:
                vm.__setattr__(attr, 999)

    def test_r1_r2(self):
        """
        Should force reflectors to be the same shape as the grid
        """
        shape = (128, 1, 64)
        vm = VM(shape=shape)

        for attr in ['rf', 'jp', 'ir', 'ij']:
            # should allow (nr, nx, ny)-sized arrays
            vm.__setattr__(attr, np.ones((1, vm.nx, vm.ny)))
            self.assertEqual(vm.nr, 1)

            # should not allow other sizes
            with self.assertRaises(ValueError) as context:
                vm.__setattr__(attr, np.ones((1, 999, vm.ny)))
            
            with self.assertRaises(ValueError) as context:
                vm.__setattr__(attr, np.ones((1, vm.nx, 999)))

    def test_get_layer_bounds(self):
        """
        Should get surfaces bounding a layer.
        """
        shape = (128, 1, 64)
        vm = VM(shape=shape)

        
        # top and bottom should be model boundary if no layers
        self.assertEqual(vm.nr, 0)

        z0, z1 = vm.get_layer_bounds(0)

        for _z in z0:
            self.assertEqual(_z, vm.r1[2])
        
        for _z in z1:
            self.assertEqual(_z, vm.r2[2])

        # bottom of first layer should be first interface
        vm.rf = 5 * np.ones((1, vm.nx, vm.ny))
        z0, z1 = vm.get_layer_bounds(0)
        
        for _z in z0:
            self.assertEqual(_z, vm.r1[2])
        
        for _z in z1:
            self.assertEqual(_z, 5)

        # top of second layer should be first interface
        z0, z1 = vm.get_layer_bounds(1)
        
        for _z in z0:
            self.assertEqual(_z, 5)
        
        for _z in z1:
            self.assertEqual(_z, vm.r2[2])


    def test_ilyr(self):
        """
        Should assign grid nodes to layers
        """
        shape = (128, 1, 64)
        vm = VM(shape=shape)

        # top layer should be 0
        for i in vm.ilyr.flatten():
            self.assertEqual(i, 0)

        # last layer should be equal to the number of interfaces 
        for nr in [1, 2, 10]:
            vm.rf = 5 * np.ones((nr, vm.nx, vm.ny))
            self.assertEqual(np.max(vm.ilyr), nr)

    def test_verify(self):
        """
        Should inherit verify functions 
        """

        shape = (128, 1, 64)
        vm = VM(shape=shape)
        vm.sl = np.ones(vm.sl.shape)

        self.assertTrue(hasattr(vm, 'verify'))
        self.assertTrue(vm.verify())


        # should return false if NaNs are on the grid
        vm.sl[0, 0, 0] = np.nan
        self.assertFalse(vm.verify())


        # should return false if zeros are on the grid
        vm.sl = np.ones(vm.sl.shape)
        self.assertTrue(vm.verify())
        vm.sl[0, 0, 0] = 0.0
        self.assertFalse(vm.verify())



    def dev_raytrace(self):
        """
        Model should be compatible with the raytracer
        """
        #XXX testing with raytracer from Rockfish, for now
        #TODO update test with PyVM raytracer, when it exists

        from rockfish.tomography.forward import raytrace_from_ascii

        temp_path = 'temp_raytrace_input'
        if os.path.isdir(temp_path):
            shutil.rmtree(temp_path)

        os.makedirs(temp_path)

        # simple vm model
        shape = (128, 1, 64)
        vm = VM(shape=shape, dx=1, dy=1, dz=1)

        vm.insert_interface(5)

        vm.define_constant_layer_velocity(0, 1500)
        vm.define_constant_layer_gradient(1, 0.2)
        
        vmfile = os.path.join(temp_path, 'model.vm')
        vm.write(vmfile)

        self.assertTrue(os.path.isfile(vmfile))

        # simple geometry and picks
        instfile = os.path.join(temp_path, 'inst.dat')
        shotfile = os.path.join(temp_path, 'shot.dat')
        pickfile = os.path.join(temp_path, 'pick.dat')

        f = open(instfile, 'w')
        f.write('100 25. 0.0 3.0\n')
        f.write('101 30. 0.0 3.0\n')
        f.close()

        f = open(shotfile, 'w')
        f.write('9000 5. 0.0 0.006\n')
        f.write('9001 10. 0.0 0.006\n')
        f.close()
        
        f = open(pickfile, 'w')
        f.write('100 9000 1 0 9.999 9.999 0.000\n')
        f.write('101 9001 1 0 9.999 9.999 0.000\n')
        f.close()


        rayfile = os.path.join(temp_path, 'out.rays')
        raytrace_from_ascii(vmfile, rayfile, instfile=instfile,
                        shotfile=shotfile, pickfile=pickfile, verbose=1000)

        self.assertTrue(os.path.isfile(rayfile))







                

def suite():
    testSuite = unittest.makeSuite(vmTestCase, 'test')
    #testSuite.addTest(doctest.DocTestSuite(vm))

    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
