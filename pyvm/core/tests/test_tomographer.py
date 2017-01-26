from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import shutil
import doctest
import unittest
import numpy as np
from pyvm.utils.loaders import get_example_file
from pyvm.models.vm import VM
from pyvm.core import tomographer
from pyvm.core.tomographer import VMTomographer
from pyvm.picks.pickdb import PickDatabase



class tomographerTestCase(unittest.TestCase):

    def test_init(self):
        """
        Should initialize a new VMTomographer instance
        """
        vmt = VMTomographer(model='benchmark2d')
        print(vmt.model)
        
        self.assertTrue(hasattr(vmt, 'pickdb'))
        self.assertTrue(isinstance(vmt.pickdb, PickDatabase))
        
        self.assertTrue(hasattr(vmt, 'model'))
        self.assertTrue(isinstance(vmt.model, VM))


    def test_pickdb(self):
        """
        Should have an attached pick database
        """
        vmt = VMTomographer(model='benchmark2d')
        
        # should add a pick
        vmt.pickdb.add_event('Pn')
        vmt.pickdb.add_source(15001, 0.0, 1.0, 2.0)
        vmt.pickdb.add_receiver(101, 0.0, 1.0, 2.0)
        vmt.pickdb.add_pick('Pn', 15001, 101, 5.34, 0.05)

        self.assertEqual(len(vmt.pickdb.picks), 1)


        


def suite():
    testSuite = unittest.makeSuite(tomographerTestCase, 'test')
    #testSuite.addTest(doctest.DocTestSuite(tomographer))

    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')


