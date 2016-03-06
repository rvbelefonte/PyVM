"""
Test suite for the pickdb module
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import doctest
import unittest
import numpy as np
from pyvm.picks import pickdb

class PickDatabaseTestCase(unittest.TestCase):

    def test_init(self):
        """
        Should initialize a basic pick database
        """
        db = pickdb.PickDatabase()
        
        # should have created basic tables and views
        self.assertTrue('events' in db.tables)
        self.assertTrue('sources' in db.tables)
        self.assertTrue('receivers' in db.tables)
        self.assertTrue('picks' in db.tables)
        self.assertTrue('master_picks' in db.views)

        # should have created VM Tomography data views
        self.assertTrue('vmtomo_receivers' in db.views)
        self.assertTrue('vmtomo_sources' in db.views)
        self.assertTrue('vmtomo_picks' in db.views)

    def test_add_event(self):
        """
        Should add a new event to the database
        """
        db = pickdb.PickDatabase()

        db.add_event('Pn')
        self.assertEqual(len(db.events), 1)

        # attempting to add a duplicate event should raise an error
        self.assertRaises(Exception, db.add_event, 'Pn') 

        # should update the existing event record
        db.add_event('Pn', branchid=99, replace=True)
        self.assertEqual(db.events['branchid'][0], 99)

    def test_add_source(self):
        """
        Should add a new source point to the database
        """
        db = pickdb.PickDatabase()

        db.add_source(15001, 0.0, 1.0, 2.0)
        self.assertEqual(len(db.sources), 1)

        # attempting to add a duplicate srcid should raise an error
        self.assertRaises(Exception, db.add_source, 15001, 99, 99, 99)

        # should update the existing record
        db.add_source(15001, 99.9, 1.0, 2.0, replace=True)
        self.assertEqual(db.sources['srcx'][0], 99.9)

    def test_add_receiver(self):
        """
        Should add a new receiver point to the database
        """
        db = pickdb.PickDatabase()

        db.add_receiver(101, 0.0, 1.0, 2.0)
        self.assertEqual(len(db.receivers), 1)

        # attempting to add a duplicate srcid should raise an error
        self.assertRaises(Exception, db.add_receiver, 101, 99, 99, 99)

        # should update the existing record
        db.add_receiver(101, 99.9, 1.0, 2.0, replace=True)
        self.assertEqual(db.receivers['recx'][0], 99.9)

    def test_add_pick(self):
        """
        Should add a new pick to the database
        """
        db = pickdb.PickDatabase()

        db.add_event('Pn')
        db.add_source(15001, 0.0, 1.0, 2.0)
        db.add_receiver(101, 0.0, 1.0, 2.0)
        db.add_pick('Pn', 15001, 101, 5.34, 0.05)
        self.assertEqual(len(db.picks), 1)

        # should raise exception if event, srcid, or recid do not exist
        self.assertRaises(Exception, db.add_pick, 'XXX', 15001, 101, 5.34, 0.05)
        self.assertRaises(Exception, db.add_pick, 'Pn', 9999, 101, 5.34, 0.05)
        self.assertRaises(Exception, db.add_pick, 'Pn', 15001, 9999, 5.34, 0.05)

    def test_to_vmtomo(self):
        """
        Should format data for VM Tomography
        """
        db = pickdb.PickDatabase()


        # add some picks
        db.add_event('Pg', branchid=2)
        db.add_event('Pn', branchid=3)

        srcids = range(15000, 15050)
        for srcid in srcids:
            db.add_source(srcid, 0.0, 1.0, 2.0)

        db.add_receiver(101, 0.0, 1.0, 2.0)
        db.add_receiver(102, 0.0, 1.0, 2.0)
        db.add_receiver(103, 0.0, 1.0, 2.0)

        db.add_pick('Pg', 15001, 101, 5.34, 0.01)
        db.add_pick('Pg', 15020, 102, 8.34, 0.03)
        db.add_pick('Pn', 15001, 101, 5.34, 0.02)
        db.add_pick('Pn', 15030, 102, 9.34, 0.04)
        db.add_pick('Pn', 15032, 101, 9.34, 0.04)
        
        # should return strings containing all data
        sources, receivers, picks = db.to_vmtomo()
        self.assertEqual(len(sources.split('\n')[0:-1]), len(db.sources))
        self.assertEqual(len(receivers.split('\n')[0:-1]), len(db.receivers))
        self.assertEqual(len(picks.split('\n')[0:-1]), len(db.picks))

        # should return strings containing only selected data
        sources, receivers, picks = db.to_vmtomo(event='Pn')
        self.assertEqual(len(picks.split('\n')[0:-1]), 3)
        self.assertEqual(len(sources.split('\n')[0:-1]), 3)
        self.assertEqual(len(receivers.split('\n')[0:-1]), 2)

        # should write data to text file
        sources_file = 'temp.sources'
        if os.path.isfile(sources_file):
            os.remove(sources_file)
        _ = db.to_vmtomo(sources_file=sources_file)
        self.assertTrue(os.path.isfile(sources_file))
        os.remove(sources_file)

        receivers_file = 'temp.receivers'
        if os.path.isfile(receivers_file):
            os.remove(receivers_file)
        _ = db.to_vmtomo(receivers_file=receivers_file)
        self.assertTrue(os.path.isfile(receivers_file))
        os.remove(receivers_file)

        picks_file = 'temp.picks'
        if os.path.isfile(picks_file):
            os.remove(picks_file)
        _ = db.to_vmtomo(picks_file=picks_file)
        self.assertTrue(os.path.isfile(picks_file))
        os.remove(picks_file)


    


def suite():
    testSuite = unittest.makeSuite(PickDatabaseTestCase, 'test')

    return testSuite

if __name__ == '__main__':
    unittest.main(defaultTest='suite')

