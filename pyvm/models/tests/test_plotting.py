"""
Test suite for the plotting module
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import doctest
import unittest
import numpy as np
from pyvm.models.vm import VM

class plottingTestCase(unittest.TestCase):

    def test_plot(self):
        """
        Should plot velocity grid with interfaces
        """
        vm = VM('benchmark2d.vm')

        vm.plot()
        

def suite():
    testSuite = unittest.makeSuite(plottingTestCase, 'test')

    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
