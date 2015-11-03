"""
Test suite for the tools module
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import doctest
import unittest
import numpy as np
from pyvm.utils.loaders import get_example_file
from pyvm.models.vm import VM

class toolsTestCase(unittest.TestCase):

    def test_insert_interface(self):
        """
        Should insert a new interface
        """
        # Initialize a new model
        vm = VM(r1=(0, 0, 0), r2=(50, 0, 30), dx=0.5, dy=0.5, dz=0.5)
        # Should add a new, flat interface at 10 km
        z0 = 10.
        vm.insert_interface(z0 * np.ones((vm.nx, vm.ny)))
        self.assertEqual(vm.nr, 1)
        self.assertEqual(vm.rf[0].min(), z0)
        self.assertEqual(vm.rf[0].max(), z0)
        # New interfaces should have jp=0
        self.assertEqual(vm.jp[0].min(), 0)
        self.assertEqual(vm.jp[0].max(), 0)
        # New layers should have ir and ij = index of new interface
        self.assertEqual(vm.ir[0].min(), 0)
        self.assertEqual(vm.ir[0].max(), 0)
        self.assertEqual(vm.ij[0].min(), 0)
        self.assertEqual(vm.ij[0].max(), 0)
        # Adding a new interface should increase ir and ij of deeper layers
        for z0 in [5., 15., 1., 20.]:
            vm.insert_interface(z0 * np.ones((vm.nx, vm.ny)))
            for iref in range(0, vm.nr):
                self.assertEqual(vm.ir[iref].min(), iref)
                self.assertEqual(vm.ir[iref].max(), iref)
                self.assertEqual(vm.ij[iref].min(), iref)
                self.assertEqual(vm.ij[iref].max(), iref)
        # should take a scalar value for a constant depth interface
        vm = VM(r1=(0, 0, 0), r2=(50, 0, 30), dx=0.5, dy=0.5, dz=0.5)
        z0 = 10.
        vm.insert_interface(z0)
        self.assertEqual(vm.nr, 1)
        self.assertEqual(vm.rf[0].min(), z0)
        self.assertEqual(vm.rf[0].max(), z0)

    def test_define_constant_layer_gradient(self):
        """
        Should define a constant layer gradient in a layer
        """
        vm = VM()

        vm.insert_interface(5)

        vm.define_constant_layer_gradient(0, 0, v0=1.500)

        self.assertAlmostEqual(vm.sl[0, 0, 0], 1. / 1.5, 7)
        
        vm.define_constant_layer_gradient(1, 0.1, v0=3.0)

        self.assertAlmostEqual(1. / np.min(vm.sl), 15.19999945, 7)

    def test_define_stretched_layer_velocities(self):
        """
        Should stretch a velocity function to fit within a layer
        """
        vm = VM()

        vm.insert_interface(5)

        vm.define_stretched_layer_velocities(0, vel=[1.500, 1.500]) 
        self.assertAlmostEqual(vm.sl[0, 0, 0], 1. / 1.5, 7)
        
        vm.define_stretched_layer_velocities(1, vel=[None, 8.1]) 

        self.assertAlmostEqual(1. / np.min(vm.sl), 8.1, 6)

    def test_define_constant_layer_velocity(self):
        """
        Should flood a layer with one velocity
        """
        vm = VM()

        vm.insert_interface(5)

        vm.define_constant_layer_velocity(0, vel=8.5)
        self.assertAlmostEqual(vm.sl[0, 0, 0], 1. / 8.5, 6)
        
        vm.define_constant_layer_velocity(1, vel=9.5)
        self.assertAlmostEqual(1. / np.min(vm.sl), 9.5, 6)
        

def suite():
    testSuite = unittest.makeSuite(toolsTestCase, 'test')

    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
