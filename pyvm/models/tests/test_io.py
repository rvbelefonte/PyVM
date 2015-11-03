"""
Test suite for the io module
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import doctest
import unittest
import numpy as np
from pyvm.utils.loaders import get_example_file
from pyvm.models.vm import VM

class ioTestCase(unittest.TestCase):

    def test_read_vm(self):
        """
        Should read the native VM format 
        """
        example_file = 'benchmark2d.vm'
        # make sure example can be found
        path = get_example_file(example_file)
        self.assertTrue(os.path.isfile(path))

        # should load the example file
        vm = VM(example_file, head_only=False)
        self.assertTrue(hasattr(vm, 'grid'))
        self.assertEqual(vm.grid.nx, 1041)
        self.assertEqual(vm.grid.ny, 1)
        self.assertEqual(vm.grid.nz, 506)
        self.assertEqual(vm.sl.shape[0], 1041)
        self.assertEqual(vm.sl.shape[1], 1)
        self.assertEqual(vm.sl.shape[2], 506)

        self.assertAlmostEqual(1. / vm.grid.values[0, 0, 0], 0.333, 3)

        self.assertEqual(vm.rf.shape, (5, vm.grid.nx, vm.grid.ny))
        self.assertEqual(vm.jp.shape, (5, vm.grid.nx, vm.grid.ny))
        self.assertEqual(vm.ir.shape, (5, vm.grid.nx, vm.grid.ny))
        self.assertEqual(vm.ij.shape, (5, vm.grid.nx, vm.grid.ny))

    def test_write_vm(self):
        """
        Should write in the native VM format
        """
        sl = np.random.rand(512, 1, 64)
        vm = VM(sl=sl)

        self.assertEqual(vm.grid.shape, sl.shape)
        self.assertAlmostEqual(vm.sl[0, 0, 0], sl[0, 0, 0], 7)

        fname = 'temp_out.vm'
        if os.path.isfile(fname):
            os.remove(fname)

        # should write to VM format
        vm.write(fname)
        self.assertTrue(os.path.isfile(fname))

        # should have the same data
        vm1 = VM(fname)
        self.assertEqual(vm1.sl.shape, vm.sl.shape)
        self.assertEqual(vm1.sl[0, 0, 0], vm.sl[0, 0, 0])
        self.assertEqual(vm1.sl[-1, -1, -1], vm.sl[-1, -1, -1])

        # clean up
        if os.path.isfile(fname):
            os.remove(fname)

    def test_write_bin(self):
        """
        Should write grid to headerless binary format
        """
        sl = np.random.rand(512, 1, 64)
        vm = VM(sl=sl)

        fname = 'temp_out.bin'
        if os.path.isfile(fname):
            os.remove(fname)

        vm.write(fname)

        f = open(fname, 'r')
        dat = np.fromstring(f.read(), dtype='float32').reshape(sl.shape)
        
        for v1, v2 in zip(dat.flatten(), vm.sl.flatten()):
            self.assertEqual(v1, v2)

        if os.path.isfile(fname):
            os.remove(fname)

def suite():
    testSuite = unittest.makeSuite(ioTestCase, 'test')

    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
