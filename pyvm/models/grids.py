"""
Utilities for managing model grids
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import copy
import numpy as np

def coord2index(x, x0, dx, nx):
    _ix = np.asarray(np.round((np.atleast_1d(x) - x0) / float(dx)),
            dtype=np.int)
    return np.clip(_ix, 0, nx - 1)

class CartesianGrid3D(object):
    """
    Class for managing a 3D cartesian grid
    """
    def __init__(self, values, origin=(0, 0, 0), spacing=(1, 1, 1)):
        """
        Class for managing a 3D cartesian grid

        Parameters
        ----------
        values: array_like
            Three-dimensional array of grid values
        origin: (x0, y0, z0), optional
            Grid origin for all 3 dimensions. Default is (0, 0, 0).
        spacing: (dx, dy, dz), optional
            Grid spacing for each for all 3 dimensions. Default is (1, 1, 1).

        Examples
        --------
        A new grid can be created by:
        >>> import numpy as np
        >>> from pyvm.models.grids import CartesianGrid3D
        >>> values = np.ones((128, 64, 128))
        >>> grd = CartesianGrid3D(values, origin=(50, 10, -5),\
                spacing=(5, 10, 5))
        >>> print(grd)
        1048576-node cartesian grid:
        x-range: [50, 685] (nx = 128, dx = 5)
        y-range: [10, 640] (ny = 64, dy = 10)
        z-range: [-5, 630] (nz = 128, dz = 5)
         values: min = 1.0, mean = 1.0, max = 1.0
        
        Standard math functions are also available:
        >>> grd *= 10.
        >>> print(grd.min(), grd.max())
        10.0 10.0
        >>> grd1 = grd.copy() * 0.5
        >>> grd /= grd1
        >>> print(grd.min(), grd.max())
        2.0 2.0
        """
        assert len(origin) == 3
        assert len(spacing) == 3

        self.values = np.atleast_3d(values)
        self.origin = np.asarray(origin)
        self.spacing = np.asarray(spacing)

    def __str__(self):

        sng = "{:}-node cartesian grid:\n"\
                .format(self.nx * self.ny * self.nz)
        sng += "x-range: [{:g}, {:g}] (nx = {:}, dx = {:g})\n"\
                .format(self.x[0], self.x[-1], self.nx, self.dx)
        sng += "y-range: [{:g}, {:g}] (ny = {:}, dy = {:g})\n"\
                .format(self.y[0], self.y[-1], self.ny, self.dy)
        sng += "z-range: [{:g}, {:g}] (nz = {:}, dz = {:g})\n"\
                .format(self.z[0], self.z[-1], self.nz, self.dz)
        sng += " values: min = {:g}, mean = {:g}, max = {:g}"\
                .format(self.min(), self.mean(), self.max())

        return sng

    def __add__(self, other):
        if isinstance(other, type(self)):
            self.values += other.values
        else:
            self.values += other

        return self

    def __sub__(self, other):
        if isinstance(other, type(self)):
            self.values -= other.values
        else:
            self.values -= other

        return self
    
    def __mul__(self, other):
        if isinstance(other, type(self)):
            self.values *= other.values
        else:
            self.values *= other
        return self

    def __truediv__(self, other):
        if isinstance(other, type(self)):
            self.values /= other.values
        else:
            self.values /= other
        return self

    def _get_dx(self):
        return self.spacing[0]

    def _set_dx(self, value):
        self.spacing[0] = value

    dx = property(fget=_get_dx, fset=_set_dx)
    
    def _get_dy(self):
        return self.spacing[1]

    def _set_dy(self, value):
        self.spacing[1] = value

    dy = property(fget=_get_dy, fset=_set_dy)

    def _get_dz(self):
        return self.spacing[2]

    def _set_dz(self, value):
        self.spacing[2] = value

    dz = property(fget=_get_dz, fset=_set_dz)

    def _get_shape(self):
        return self.values.shape
    shape = property(fget=_get_shape)

    def _get_nx(self):
        return self.shape[0]
    nx = property(fget=_get_nx)

    def _get_ny(self):
        return self.shape[1]
    ny = property(fget=_get_ny)

    def _get_nz(self):
        return self.shape[2]
    nz = property(fget=_get_nz)

    def _get_x(self):
        return self.origin[0] + self.dx * np.arange(self.nx)
    x = property(fget=_get_x)
    
    def _get_y(self):
        return self.origin[1] + self.dy * np.arange(self.ny)
    y = property(fget=_get_y)
    
    def _get_z(self):
        return self.origin[2] + self.dz * np.arange(self.nz)
    z = property(fget=_get_z)

    def x2i(self, x):
        """
        Convert x coordinates to indices
        """
        return coord2index(x, self.origin[0], self.dx, self.nx)
    
    def y2i(self, y):
        """
        Convert y coordinates to indices
        """
        return coord2index(y, self.origin[1], self.dy, self.ny)

    def z2i(self, z):
        """
        Convert z coordinates to indices
        """
        return coord2index(z, self.origin[2], self.dz, self.nz)

    def xyz2ijk(self, coords):
        """
        Convert coordinates to indices

        Parameters
        ----------
        coords: array_like
            Array of (x, y, z) coordinates

        Returns
        -------
        indices: ndarray
            Array of (ix, iy, iz) indices
        """
        coords = np.asarray(coords)

        ix = self.x2i(coords[:, 0])
        iy = self.y2i(coords[:, 1])
        iz = self.z2i(coords[:, 2])

        return np.asarray([ix, iy, iz]).T

    def min(self, **kwargs):
        """
        Return the minimum of the grid or minimum along an axis

        See :meth:`numpy.min` for keyword arguments.
        """
        return np.min(self.values, **kwargs)

    def max(self, **kwargs):
        """
        Return the max of the grid or minimum along an axis

        See :meth:`numpy.max` for keyword arguments.
        """
        return np.max(self.values, **kwargs)

    def mean(self, **kwargs):
        """
        Return the mean of the grid or minimum along an axis

        See :meth:`numpy.mean` for keyword arguments.
        """
        return np.mean(self.values, **kwargs)

    def copy(self):
        """
        Make a copy of the grid instance
        """
        return copy.deepcopy(self)

    def to_bin(self, path_or_buf=None, order=(0, 1, 2),
            dtype=np.float32):
        """
        Write grid values to a headerless binary file

        Parameters
        ----------
        path_or_buf: string, file handle, or None
            File path or object to write data to. If None (default), the
            string representation of the data is returned.
        order: list, optional
            List defining the order of grid dimensions in the output.
            Default is `(0, 1, 2)` (i.e., x, y, z).
        dtype: data-type, optional
            The desired data type for the array, e.g., `numpy.int8`.
            Default is `numpy.float32`.
        """
        assert len(order) == len(self.shape),\
                'order must have len = {:}'.format(len(self.shape))

        gridsize0 = self.shape
        gridsize1 = tuple([gridsize0[i] for i in order])
        dat = self.values.reshape(gridsize1)

        dat = np.asarray(dat, dtype=dtype).tostring()

        if path_or_buf is None:
            return dat
        elif hasattr(path_or_buf, 'write'):
            path_or_buf.write(dat) 
        else:
            f = open(path_or_buf, 'w')
            f.write(dat)
            f.close()
