"""
Tools for working with VM models
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import numpy as np
from scipy.interpolate import interp1d


class VMTools(object):
    """
    Convience class for VM model tools 
    """
    def insert_interface(self, rf, jp=None, ir=None, ij=None):
        """
        Insert a new interface into the model.

        Parameters
        ----------
        rf : {scalar, array_like}
            Constant depth or matrix of depths with shape ``(nx, ny)``.
        jp : {scalar, array_like, None}, optional
            Constant slowness jumps, matrix of depths with shape
            ``(nx, ny)``, or None. If None (default), jumps are set to zero.
        ir : {scalar, array_like, None}, optional
            Sets indices of the interface to use in the inversion
            for interface depths at each interface node. A value of ``-1``
            indicates that a node should not be used in the inversion. 
            Can be a scalar value that is copied to all nodes, a matrix of
            with shape ``(nx, ny)``, or None. If None (default), all
            values are set to the index of the new interface.
        ij : {scalar, array_like, None}, optional
            Sets indices of the interface to use in the inversion
            for slowness jumps at each interface node. A value of ``-1``
            indicates that a node should not be used in the inversion. 
            Can be a scalar value to be copied to all nodes, a matrix
            with shape ``(nx, ny)``, or None. If None (default), all
            values are set to the index of the new interface.

        Returns
        -------
        iref : int
            Index of the new interface.
        """
        # Expand scalar depth value to array
        if np.asarray(rf).size == 1:
            rf *= np.ones((self.nx, self.ny))
        # Determine index for new interface
        iref = 0
        for _iref in range(0, self.nr):
            if np.nanmax(rf) >= np.nanmax(self.rf[_iref]):
                iref += 1
        # Create default arrays if not given
        if jp is None:
            jp = np.zeros((self.nx, self.ny))
        if ir is None:
            ir = iref * np.ones((self.nx, self.ny))
        if ij is None:
            ij = iref * np.ones((self.nx, self.ny))
        # Expand jump and flag arrays if scalars
        if np.asarray(jp).size == 1:
            jp *= np.ones((self.nx, self.ny))
        if np.asarray(ir).size == 1:
            ir *= np.ones((self.nx, self.ny))
        if np.asarray(ij).size == 1:
            ij *= np.ones((self.nx, self.ny))
        # Check for proper dimensions
        for v in [rf, jp, ir, ij]:
            assert np.asarray(v).shape == (self.nx, self.ny),\
                'Arrays must have shape (nx, ny)'
        # Insert arrays for new interface
        if self.nr == 0:
            self.rf = np.asarray([rf])
            self.jp = np.asarray([jp])
            self.ir = np.asarray([ir])
            self.ij = np.asarray([jp])
        else:
            self.rf = np.insert(self.rf, iref, rf, 0)
            self.jp = np.insert(self.jp, iref, jp, 0)
            self.ir = np.insert(self.ir, iref, ir, 0)
            self.ij = np.insert(self.ij, iref, ij, 0)
        for _iref in range(iref + 1, self.nr):
            idx = np.nonzero(self.ir[_iref] >= iref)
            self.ir[_iref][idx] += 1
            idx = np.nonzero(self.ij[_iref] >= iref)
            self.ij[_iref][idx] += 1
        return iref

    def define_constant_layer_gradient(self, ilyr, dvdz, v0=None, xmin=None,
                                       xmax=None, ymin=None, ymax=None):
        """
        Replace velocities within a layer by defining a constant gradient.

        Parameters
        ----------
        ilyr: int
            Index of layer to work on.
        dvdz: float
            Velocity gradient.
        v0: float
            Velocity at the top of the layer. Default is to use the 
            value at the base of the overlying layer.
        xmin, xmax: float
            Set the x-coordinate limits for modifying velocities. Default
            is to change velocities over the entire x-domain.
        ymin, ymax: float
            Set the y-coordinate limits for modifying velocities. Default is
            to change velocities over the entire y-domain.
        """
        z0, z1 = self.get_layer_bounds(ilyr)
        for ix in self.xrange2i(xmin, xmax):
            for iy in self.yrange2i(ymin, ymax):
                iz0, iz1 = self.grid.z2i((z0[ix, iy], z1[ix, iy]))
                iz1 += 1
                z = self.grid.z[iz0:iz1] - self.grid.z[iz0]
                if v0 is None:
                    if iz0 == 0:
                        _v0 = 0.
                    else:
                        _v0 = 1. / self.sl[ix, iy, iz0 - 1]
                else:
                    _v0 = v0
                self.sl[ix, iy, iz0:iz1] = 1. / (_v0 + z * dvdz)

    def define_stretched_layer_velocities(self, ilyr, vel=[None, None],
                                          xmin=None, xmax=None, ymin=None,
                                          ymax=None, kind='linear'):
        """
        Define velocities within a layer by stretching a velocity function.

        At each x,y position in the model grid, the list of given velocities
        are first distributed proportionally across the height of the layer.
        Then, this z,v function is used to interpolate velocities for each
        depth node in the layer.

        Parameters
        ----------
        ilyr: int
            Index of layer to work on.
        vel: array_like, optional
            List of layer velocities. Default is to stretch a 1d function
            between the deepest velocity of the overlying layer and the
            shallowest velocity of the underlying layer.
        xmin, xmax: float
            Set the x-coordinate limits for modifying velocities. Default is
            to change velocities over the entire x-domain.
        ymin, ymax: float
            Set the y-coordinate limits for modifying velocities. Default is
            to change velocities over the entire y-domain.
        kind: str or int, optional
            Specifies the kind of interpolation as a string ('linear',
            'nearest', 'zero', 'slinear', 'quadratic, 'cubic') or as an
            integer specifying the order of the spline interpolator to use.
            Default is 'linear'.
        """
        # Get layer boundary depths
        z0, z1 = self.get_layer_bounds(ilyr)
        # Define velocity function at each x,y point
        vel = np.asarray(vel)
        nvel = len(vel)
        for ix in self.xrange2i(xmin, xmax):
            for iy in self.yrange2i(ymin, ymax):
                iz0, iz1 = self.grid.z2i((z0[ix, iy], z1[ix, iy]))
                iz1 += 1
                z = self.grid.z[iz0:iz1]
                if len(z) == 0:
                    # in pinchout, nothing to do
                    continue
                # Make a copy of velocities for this iteration
                _vel = np.copy(vel)
                # Get top and bottom velocities, if they are None
                if vel[0] is None:
                    _vel[0] = 1. / self.sl[ix, iy, max(iz0 - 1, 0)]
                if len(vel) > 1:
                    if vel[1] is None:
                        _vel[1] = 1. / self.sl[ix, iy, min(iz1 + 1, self.nx)]
                # Pad interpolates for rounding to grid coordinates
                if nvel == 1:
                    # Set constant value
                    v = np.asarray([_vel])
                else:
                    # Interpolate velocities
                    zi = z0[ix, iy] + (z1[ix, iy] - z0[ix, iy])\
                            * np.arange(0., nvel) / (nvel - 1)
                    vi = np.copy(_vel)
                    if z[0] < zi[0]:
                        zi = np.insert(zi, 0, z[0])
                        vi = np.insert(vi, 0, _vel[0])
                    if z[-1] > zi[-1]:
                        zi = np.append(zi, z[-1])
                        vi = np.append(vi, _vel[-1])
                    zv = interp1d(zi, vi, kind=kind)
                    v = zv(z)
                self.sl[ix, iy, iz0:iz1] = 1. / v

    def define_constant_layer_velocity(self, ilyr, vel, xmin=None,
                                          xmax=None, ymin=None, ymax=None):
        """
        Define a constant velocity for an entire layer.

        ilyr: int
            Index of layer to work on.
        vel: float
            Velocity value to flood the layer with.
        xmin, xmax: float
            Set the x-coordinate limits for modifying velocities. Default is
            to change velocities over the entire x-domain.
        ymin, ymax: float
            Set the y-coordinate limits for modifying velocities. Default is
            to change velocities over the entire y-domain.
        """
        self.define_stretched_layer_velocities(ilyr, [vel], xmin=xmin,
                xmax=xmax, ymin=ymin, ymax=ymax)
