"""
Raytracing utilities
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import subprocess
import warnings
import time
import tempfile

def _pickarray2tables(pickdat):
    """
    Builds source, receiver, and pick tables from an array of pick times.
    
    Parameters
    ----------
    pickdat: array_like
         9xN array with rows containing [src_x, srx_y, src_z, rec_x, rec_y,
         src_z, pick_time, phase_id, branch_id].

    Returns
    -------
    src: numpy.ndarray
        4xN array with rows containing [src_id, src_x, src_y, src_z]
    rec: numpy.ndarray
        4xN array with rows containing [rec_id, rec_x, rec_y, rec_z]
    picks: numpy.ndarray
        7xN array with rows containing [rec_id, src_id, phase_id, branch_id, _x, rec_y, rec_z]
    """
f.write('100 9000 1 0 9.999 9.999 0.000\n')


        sql = 'SELECT DISTINCT ensemble, trace, branch, subbranch, offset,'
        sql += 'time, error FROM {:}'.format(MASTER_VIEW)


def _raytrace_from_arrays(vmfile, pickdat, warnonly=True):
    """
    Backend for the raytracer that takes pick data as an array

    Parameters
    ----------
    vmfile: str
        Filename of a VM-format velocity model.

    pickdat: array_like
         9xN array with rows containing [src_x, srx_y, src_z, rec_x, rec_y,
         src_z, pick_time, phase_id, branch_id].
    """
    srcdat = [[d[0], d[1], d[2]] for d in pickdat]
    recdat = [[d[3], d[4], d[5]] for d in pickdat]
    recdat = np.unique([[d[3], d[4], d[5]] for d in pickdat])


    #instfile = tempfile.NamedTemporaryFile(delete=True)
    #shotfile = tempfile.NamedTemporaryFile(delete=True)
    #pickfile = tempfile.NamedTemporaryFile(delete=True)

    


    raise NotImplementedError

f = open(instfile, 'w')
f.write('100 25. 0.006 0.0\n')
f.close()

f = open(shotfile, 'w')
f.write('9000 5. 0.0 0.006\n')
f.close()

f = open(pickfile, 'w')
f.write('100 9000 1 0 9.999 9.999 0.000\n')
f.close()
