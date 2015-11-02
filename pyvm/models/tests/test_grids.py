"""
Test suite for the grids module
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import doctest
import unittest
import numpy as np
from pyvm.models import grids
from pyvm.models.grids import CartesianGrid3D


class gridsTestCase(unittest.TestCase):
    """
    Test cases for the VMGrid class
    """
    def test_init(self):
        """
        Should initialize a new CartesianGrid3D object
        """
        # should create new grid
        values = np.zeros((51, 1, 51))
        grd = CartesianGrid3D(values)

        # should have dimension attributes
        self.assertEqual(grd.nx, values.shape[0])
        self.assertEqual(grd.ny, values.shape[1])
        self.assertEqual(grd.nz, values.shape[2])

        # should have default grid spacing of '1'
        self.assertEqual(grd.dx, 1)
        self.assertEqual(grd.dy, 1)
        self.assertEqual(grd.dz, 1)
        
        # should have default origin '0'
        self.assertEqual(grd.origin[0], 0)
        self.assertEqual(grd.origin[1], 0)
        self.assertEqual(grd.origin[1], 0)

        # should always create a 3D grid
        values = np.zeros((51, 51))
        grd = CartesianGrid3D(values)

        self.assertEqual(grd.nx, values.shape[0])
        self.assertEqual(grd.ny, values.shape[1])
        self.assertEqual(grd.nz, 1)

    def test_xyz(self):
        """
        Should generate arrays of coordinates
        """
        values = np.zeros((10, 20, 30))
        grd = CartesianGrid3D(values)

        self.assertEqual(grd.x[-1], 9)
        self.assertEqual(grd.y[-1], 19)
        self.assertEqual(grd.z[-1], 29)

        self.assertEqual(len(grd.x), grd.nx)
        self.assertEqual(len(grd.y), grd.ny)
        self.assertEqual(len(grd.z), grd.nz)

    def test_x2i(self):
        """
        Should convert coordinates to indices
        """
        values = np.zeros((10, 20, 30))
        grd = CartesianGrid3D(values)

        # should work on individual dimensions
        self.assertEqual(grd.x2i(grd.x[-1]), grd.nx - 1)
        self.assertEqual(grd.y2i(grd.y[-1]), grd.ny - 1)
        self.assertEqual(grd.z2i(grd.z[-1]), grd.nz - 1)

        # should work on lists of coordinates
        coords = [(0, 0, 0), (9, 19, 29)]
        ijk = grd.xyz2ijk(coords)

        self.assertEqual(ijk[0, 0], 0)
        self.assertEqual(ijk[0, 1], 0)
        self.assertEqual(ijk[0, 2], 0)
        self.assertEqual(ijk[1, 0], grd.nx - 1)
        self.assertEqual(ijk[1, 1], grd.ny - 1)
        self.assertEqual(ijk[1, 2], grd.nz - 1)

    def test_add(self):
        """
        Should add values to the grid
        """
        values = np.zeros((51, 1, 51))
        grd = CartesianGrid3D(values)

        # should add scalars
        grd += 1
        self.assertEqual(grd.max(), 1)

        # should add other identical grids
        grd2 = grd.copy() * 2.
        grd += grd2
        self.assertEqual(grd.max(), 3.)

    def test_sub(self):
        """
        Should add values to the grid
        """
        values = np.zeros((51, 1, 51))
        grd = CartesianGrid3D(values)

        # should subtract scalars
        grd -= 1
        self.assertEqual(grd.max(), -1)

        # should subtract other identical grids
        grd2 = grd.copy() + 2.
        grd -= grd2
        self.assertEqual(grd.max(), -2.)

    def test_mul(self):
        """
        Should multiply the grid by values 
        """
        values = np.ones((51, 1, 51))
        grd = CartesianGrid3D(values)

        # should multiply scalars
        grd *= 10.
        self.assertEqual(grd.max(), 10.)

        # should multiply other identical grids
        grd2 = grd.copy() + 2.
        grd *= grd2
        self.assertEqual(grd.max(), 120.)

    def test_div(self):
        """
        Should divide the grid by values 
        """
        values = np.ones((51, 1, 51))
        grd = CartesianGrid3D(values)

        # should multiply scalars
        grd /= 10. # float division
        self.assertEqual(grd.max(), 0.1)
        grd /= 10 # integer division should use future division
        self.assertEqual(grd.max(), 0.01)

        # should multiply other identical grids
        grd2 = grd.copy() * 100.
        grd /= grd2
        self.assertEqual(grd.max(), 0.01)

    def test_numpy_funcs(self):
        """
        Should pass the grid to numpy functions
        """
        values = np.ones((51, 1, 51))
        grd = CartesianGrid3D(values)

        self.assertEqual(grd.min(), 1)
        self.assertEqual(grd.max(), 1)
        self.assertEqual(grd.mean(), 1)

        grd *= 10.
        
        self.assertEqual(grd.min(), 10.)
        self.assertEqual(grd.max(), 10.)
        self.assertEqual(grd.mean(), 10.)

    def test_to_bin(self):
        values = np.random.rand(51, 1, 51)
        grd = CartesianGrid3D(values)

        # should return a string
        dat = grd.to_bin()
        values2 = np.fromstring(dat, dtype='float32').reshape((grd.nx,
            grd.ny, grd.nz))
        for v1, v2 in zip(values.flatten(), values2.flatten()):
            self.assertAlmostEqual(v1, v2, 5)

        # should write to a file
        fname = 'temp.bin'
        if os.path.isfile(fname):
            os.remove(fname)

        grd.to_bin(fname)
        self.assertTrue(os.path.isfile(fname))

        f = open(fname, 'r')
        dat = np.fromstring(f.read(), dtype='float32').reshape((grd.nx,
            grd.ny, grd.nz))
        for v1, v2 in zip(values.flatten(), values2.flatten()):
            self.assertAlmostEqual(v1, v2, 5)
            
        os.remove(fname)

        # should write to a buffer
        fname = 'temp.bin'
        if os.path.isfile(fname):
            os.remove(fname)

        f = open(fname, 'w')
        grd.to_bin(f)
        f.close()

        f = open(fname, 'r')
        dat = np.fromstring(f.read(), dtype='float32').reshape((grd.nx,
            grd.ny, grd.nz))
        for v1, v2 in zip(values.flatten(), values2.flatten()):
            self.assertAlmostEqual(v1, v2, 5)
            
        os.remove(fname)


def suite():
    testSuite = unittest.makeSuite(gridsTestCase, 'test')
    testSuite.addTest(doctest.DocTestSuite(grids))

    return testSuite


def suite():
    testSuite = unittest.makeSuite(gridsTestCase, 'test')
    testSuite.addTest(doctest.DocTestSuite(grids))

    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
