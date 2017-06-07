"""
Test suite for the rayfan module
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import doctest
import unittest
import numpy as np
from pyvm.utils.loaders import get_example_file
from pyvm.forward.raytracing import rayfan


class rayfanTestCase(unittest.TestCase):

    def test_read(self):
        """
        Should read a rayfan group file from the disk
        """
        rays = rayfan.readRayfanGroup(
            get_example_file('test1.rays'))

        self.assertEqual(len(rays.rayfans), 2)

    def test_write(self):
        """
        Should write a rayfan group file to the disk
        """
        tempfile = 'temp123.rays'
        if os.path.isfile(tempfile):
            os.remove(tempfile)

        # read example rayfan group
        rays = rayfan.readRayfanGroup(
            get_example_file('test1.rays'))

        # should write a copy to the disk
        rays.write(tempfile)
        self.assertTrue(os.path.isfile(tempfile))

        # should have correct data
        rays1 = rayfan.readRayfanGroup(tempfile)

        self.assertEqual(len(rays.rayfans), len(rays1.rayfans))

        for az0, az1 in zip(rays.azimuths, rays1.azimuths):
            self.assertEqual(az0, az1)

        for r0, r1 in zip(rays.offsets, rays1.offsets):
            self.assertEqual(r0, r1)

        for e0, e1 in zip(rays.residuals, rays1.residuals):
            self.assertEqual(e0, e1)

        for rfn0, rfn1 in zip(rays.rayfans, rays1.rayfans):

            # should have correct metadata
            self.assertEqual(rfn0.start_point_id,
                             rfn1.start_point_id)
            
            self.assertEqual(rfn0.nrays, rfn1.nrays)
            
            # should have correct number of raypaths
            self.assertEqual(len(rfn0.paths), len(rfn1.paths))

            for i in range(len(rfn0.paths)):
                # should have correct path metadata
                self.assertEqual(rfn0.end_point_ids[i], rfn1.end_point_ids[i])
                self.assertEqual(rfn0.event_ids[i], rfn1.event_ids[i])
                self.assertEqual(rfn0.event_subids[i], rfn1.event_subids[i])
                self.assertEqual(rfn0.pick_times[i], rfn1.pick_times[i])
                self.assertEqual(rfn0.travel_times[i], rfn1.travel_times[i])
                self.assertEqual(rfn0.pick_errors[i], rfn1.pick_errors[i])

                for j in range(len(rfn0.paths[i])):
                    for k in range(3):
                        # should have correct path coordinates
                        self.assertEqual(rfn0.paths[i][j][k], rfn1.paths[i][j][k])

        if os.path.isfile(tempfile):
            os.remove(tempfile)


def suite():
    testSuite = unittest.makeSuite(rayfanTestCase, 'test')
    #testSuite.addTest(doctest.DocTestSuite(vrayfanm))

    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
