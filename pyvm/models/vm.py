"""
Utilities for working with VM Tomography models


Examples
--------
Load an example model:
>>> from pyvm.models.vm import VM
>>> vm = VM('benchmark2d.vm')
>>> print(vm.__str__(extended=True))
============================Slowness Model============================
526746-node cartesian grid:
x-range: [0, 260] (nx = 1041, dx = 0.25)
y-range: [0, 0] (ny = 1, dy = 1)
z-range: [-0.5, 50] (nz = 506, dz = 0.1)
 values: min = 0.120904, mean = 0.177293, max = 3.003
============================Slowness Model============================
Model top: z = -0.5
 Layer 0: u = [  0.120,   3.003] (v = [  0.333,   8.347])
Interface 0: z = [-0.439999997616, 0.0]
 Layer 1: u = [  0.181,   3.003] (v = [  0.333,   5.517])
Interface 1: z = [-0.340000003576, 1.52999997139]
 Layer 2: u = [  0.179,   3.003] (v = [  0.333,   5.586])
Interface 2: z = [-0.239999994636, 3.08959817886]
 Layer 3: u = [  0.161,   0.667] (v = [  1.499,   6.225])
Interface 3: z = [15.0, 27.0798377991]
 Layer 4: u = [  0.120,   0.155] (v = [  6.436,   8.350])
Interface 4: z = [40.0, 40.0]
 Layer 5: u = [  0.120,   0.125] (v = [  7.987,   8.305])
Model bottom: z = 50.0000007525

Create a new model:
>>> from pyvm.models.vm import VM
>>> vm = VM(shape=(512, 1, 512), spacing=(0.5, 1, 0.1),\
        origin=(-0.5, 0, 35))
>>> vm.sl = 1. / (1500. * np.ones(vm.grid.shape))
>>> vm.insert_interface(-0.1)
0
>>> vm.insert_interface(5)
1
>>> print(vm)
============================Slowness Model============================
262144-node cartesian grid:
x-range: [-0.5, 255] (nx = 512, dx = 0.5)
y-range: [0, 0] (ny = 1, dy = 1)
z-range: [35, 86.1] (nz = 512, dz = 0.1)
 values: min = 0.000666667, mean = 0.000666667, max = 0.000666667
============================Slowness Model============================
[Use 'print(VM.__str__(extended=True))' for more detailed information]
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import copy
import numpy as np
import warnings
from pyvm.utils.string_tools import pad_string
from pyvm.models.grids import CartesianGrid3D
from pyvm.models.io import VMIO
from pyvm.models.tools import VMTools
from pyvm.models.plotting import VMPlotter


class VM(VMIO, VMTools, VMPlotter):

    def __init__(self, filename=None, **kwargs):
        """
        Class for managing a VM Tomography model
        """
        if filename is not None:
            self.read(filename, **kwargs)
        else:
            shape = kwargs.pop('shape', (128, 1, 128))
            self._init_model(shape, **kwargs)

    def __str__(self, extended=False, title='Slowness Model'):
        """
        Print an overview of the VM model.

        Parameters
        ----------
        extended: bool, optional
            Determines whether or not to print detailed information about
            each layer. Default is to print an overview.
        title: str, optional
            Sets the title in the banner. Default is 'Slowness Model'.
        """
        banner = pad_string(title, char='=', width=70)
        sng = banner + '\n'
        sng += self.grid.__str__() 
        sng += '\n' + banner
        if not extended:
            sng += "\n[Use 'print(VM.__str__(extended=True))' for"
            sng += " more detailed information]"

        else:
            self.apply_jumps()
            sng += '\nModel top: z = {:}\n'.format(self.r1[2])
            sng += ' ' + self._format_layer_info(0) + '\n'
            for iref in range(0, self.nr):
                sng += 'Interface {:}: z = [{:}, {:}]\n'\
                        .format(iref, np.min(self.rf[iref]),
                                np.max(self.rf[iref]))
                sng += ' ' + self._format_layer_info(iref + 1) + '\n'
            sng += 'Model bottom: z = {:}'.format(self.r2[2])
            self.remove_jumps()

        return sng

    def _format_layer_info(self, ilyr):

        idx = np.nonzero(self.ilyr.flatten() == ilyr)
        sl = self.sl.flatten()
        smin = np.min(sl[idx])
        smax = np.max(sl[idx])
        sng = 'Layer {:}: u = [{:7.3f}, {:7.3f}]'\
                .format(ilyr, smin, smax)
        sng += ' (v = [{:7.3f}, {:7.3f}])'.format(1. / smax, 1. / smin)
        
        return sng

    def _init_model(self, shape, origin=(0, 0, 0),
            spacing=(1, 1, 1), dtype=np.float32, **kwargs):
        """
        (Re)initialize the model domain.

        Parameters
        ----------
        shape: tuple
            Tuple of (nx, ny, nz) model grid dimensions
        origin: tuple, optional
            Grid orgin given as '(x0, y0, z0)'. Default is '(0, 0, 0)'.
        spacing: tuple, optional
            Grid spacing given as '(dx, dy, dz)'. Default is '(1, 1, 1)'.
        dtype: data-type, optional
            The desired data type for the grid, e.g., `numpy.int8`.
            Default is `numpy.float32`.
        """
        shape = np.asarray(shape)
        shape[0] = kwargs.pop('nx', shape[0])
        shape[1] = kwargs.pop('ny', shape[1])
        shape[2] = kwargs.pop('nz', shape[2])

        values = np.asarray(kwargs.pop('sl', np.zeros(shape)),
                dtype=dtype)

        self.grid = CartesianGrid3D(values, origin=origin, spacing=spacing)

        self.rf = []
        self.jp = []
        self.ir = []
        self.ij = []

        for attr in ['dx', 'dy', 'dz', 'sl', 'rf', 'jp', 'ir', 'ij', 'r1',
                'r2']:
            if attr in kwargs:
                self.__setattr__(attr, kwargs[attr])

    def _get_sl(self):
        return self.grid.values

    def _set_sl(self, values):
        self.grid.values = values

    sl = property(fget=_get_sl, fset=_set_sl)

    def _get_nx(self):
        return self.grid.nx

    nx = property(fget=_get_nx)
    
    def _get_ny(self):
        return self.grid.ny

    ny = property(fget=_get_ny)
    
    def _get_nz(self):
        return self.grid.nz

    nz = property(fget=_get_nz)

    def _get_dx(self):
        return self.grid.spacing[0]
    
    def _set_dx(self, value):
        self.grid.spacing[0] = value

    dx = property(fget=_get_dx, fset=_set_dx)

    def _get_dy(self):
        return self.grid.spacing[1]
    
    def _set_dy(self, value):
        self.grid.spacing[1] = value

    dy = property(fget=_get_dy, fset=_set_dy)
    
    def _get_dz(self):
        return self.grid.spacing[2]
    
    def _set_dz(self, value):
        self.grid.spacing[2] = value

    dz = property(fget=_get_dz, fset=_set_dz)

    def _get_r1(self):
        return self.grid.origin

    def _set_r1(self, value):
        if len(value) != 3:
            raise ValueError('r1 must have 3 elements: (x0, y0, z0)')
        self.grid.origin = value

    r1 = property(fget=_get_r1, fset=_set_r1)

    def _get_r2(self):
        return np.asarray([self.grid.x[-1], self.grid.y[-1],
            self.grid.z[-1]])

    def _set_r2(self, value):
        if len(value) != 3:
            raise ValueError('r2 must have 3 elements: (x0, y0, z0)')

        self.dx = (value[0] - self.grid.origin[0]) / self.grid.nx
        self.dy = (value[1] - self.grid.origin[1]) / self.grid.ny
        self.dz = (value[2] - self.grid.origin[2]) / self.grid.nz

    r2 = property(fget=_get_r2, fset=_set_r2)

    def _get_nr(self):
        return len(self.rf)

    nr = property(fget=_get_nr)

    def _get_rf(self):
        return self._rf

    def _set_rf(self, value):
        val = np.atleast_1d(value)

        if len(val) == 0:
            self._rf = val
            return

        if (val.shape[1], val.shape[2])\
                != (self.nx, self.ny):
            raise ValueError('shape of rf must be (*, nx, ny)')
        else:
            self._rf = val

    rf = property(fget=_get_rf, fset=_set_rf)

    def _get_jp(self):
        return self._jp

    def _set_jp(self, value):
        val = np.atleast_1d(value)

        if len(val) == 0:
            self._jp = val
            return

        if (val.shape[1], val.shape[2])\
                != (self.nx, self.ny):
            raise ValueError('shape of jp must be (*, nx, ny)')
        else:
            self._jp = val

    jp = property(fget=_get_jp, fset=_set_jp)

    def _get_ir(self):
        return self._ir

    def _set_ir(self, value):
        val = np.atleast_1d(value)

        if len(val) == 0:
            self._ir = val
            return

        if (val.shape[1], val.shape[2])\
                != (self.nx, self.ny):
            raise ValueError('shape of ir must be (*, nx, ny)')
        else:
            self._ir = val

    ir = property(fget=_get_ir, fset=_set_ir)

    def _get_ij(self):
        return self._ij

    def _set_ij(self, value):
        val = np.atleast_1d(value)

        if len(val) == 0:
            self._ij = val
            return

        if (val.shape[1], val.shape[2])\
                != (self.nx, self.ny):
            raise ValueError('shape of ij must be (*, nx, ny)')
        else:
            self._ij = val

    ij = property(fget=_get_ij, fset=_set_ij)

    def _get_ilyr(self):
        """
        Returns a grid of layer indices for each node in the slowness grid.
        """
        lyr = np.zeros((self.nx, self.ny, self.nz))
        for iref in range(self.nr + 1):
            z0, z1 = self.get_layer_bounds(iref)
            for ix in range(0, self.nx):
                for iy in range(0, self.ny):
                    zrange = np.asarray(z0[ix, iy], z1[ix, iy], self.dz)
                    iz = self.grid.z2i(zrange)
                    lyr[ix, iy, iz] = iref
        return lyr

    ilyr = property(fget=_get_ilyr)

    def copy(self):
        """
        Returns a copy of the model instance
        """
        return copy.deepcopy(self)

    def xrange2i(self, xmin=None, xmax=None):
        """
        Returns a list of x indices for a given x range.

        Parameters
        ----------
        xmin: float, optional
            Minimum value of x. Default is the minimum x value in the model.
        xmax: float, optional
            Maximum value of x. Default is the maximum x value in the model.

        Returns
        -------
        ix: ndarray
            Array of x indices
        """
        if xmin is None:
            xmin = self.r1[0]
        else:
            xmin = max(self.r1[0], xmin)
        if xmax is None:
            xmax = self.r2[0]
        else:
            xmax = min(self.r2[0], xmax)

        return range(self.grid.x2i([xmin])[0], self.grid.x2i([xmax])[0] + 1)

    def yrange2i(self, ymin=None, ymax=None):
        """
        Returns a list of y indices for a given y range.

        Parameters
        ----------
        ymin: float, optional
            Minimum value of y. Default is the minimum y value in the model.
        ymax: float, optional
            Maximum value of y. Default is the maximum y value in the model.

        Returns
        -------
        iy: ndarray
            Array of y indices
        """
        if ymin is None:
            ymin = self.r1[1]
        else:
            ymin = max(self.r1[1], ymin)
        if ymax is None:
            ymax = self.r2[1]
        else:
            ymax = min(self.r2[1], ymax)

        return range(self.grid.y2i([ymin])[0], self.grid.y2i([ymax])[0] + 1)

    def zrange2i(self, zmin=None, zmax=None):
        """
        Returns a list of z indices for a given z range.

        Parameters
        ----------
        zmin: float, optional
            Minimum value of z. Default is the minimum y value in the model.
        zmax: float, optional
            Maximum value of z. Default is the maximum z value in the model.

        Returns
        -------
        iz: ndarray
            Array of z indices
        """

        if zmin is None:
            zmin = self.r1[2]
        else:
            zmin = max(self.r1[2], zmin)
        if zmax is None:
            zmax = self.r2[2]
        else:
            zmax = min(self.r2[2], zmax)

        return range(self.grid.z2i([zmin])[0], self.grid.z2i([zmax])[0] + 1)

    def get_layer_bounds(self, ilyr):
        """
        Get surfaces bounding a layer.

        Parameter
        ---------
        ilyr: int
            Index of layer of interest.

        Returns
        -------
        z0, z1
            arrays of top, bottom interface depths
        """
        assert (ilyr >= 0) and (ilyr <= self.nr),\
            "Layer {:} does not exist.".format(ilyr)
        if ilyr == 0:
            z0 = np.ones((self.nx, self.ny)) * self.r1[2]
        else:
            z0 = self.rf[ilyr - 1]
        if ilyr >= self.nr:
            z1 = np.ones((self.nx, self.ny)) * self.r2[2]
        else:
            z1 = self.rf[ilyr]
        return z0, z1

    def apply_jumps(self, iref=None, remove=False):
        """
        Apply slowness jumps to the grid.

        Parameters
        ----------
        iref: array_like, optional
            List of interface indices to apply jumps at. Default is to
            apply jumps at all interfaces.
        remove: bool, optional
            Determines whether jumps should be removed  or applied. Default is
            `False` (i.e., jumps are applied).
        """
        if iref is None:
            iref = range(0, self.nr)
        for _iref in iref:
            z0, _ = self.get_layer_bounds(_iref + 1)
            for ix in range(0, self.nx):
                for iy in range(0, self.ny):
                    iz0 = self.grid.z2i([z0[ix, iy]])[0]
                    if remove is False:
                        self.sl[ix, iy, iz0:] += self.jp[_iref, ix, iy]
                    else:
                        self.sl[ix, iy, iz0:] -= self.jp[_iref, ix, iy]

    def remove_jumps(self, iref=None):
        """
        Remove slowness jumps from the grid.
       
        iref: array_like, optional
            List of interface indices to remove jumps at. Default is to
            remove jumps at all interfaces.
        """
        self.apply_jumps(iref=iref, remove=True)
