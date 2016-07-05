"""
Tools for working with VM models
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d


def gauss_kern(size, sizey=None):
    """
    Returns a normalized 2D gauss kernel array for convolutions
    """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2 / float(size) + y**2 / float(sizey)))

    return g / g.sum()


def smooth2d(im, n, ny=None):
    """
    Blurs the image by convolving with a gaussian kernel of typical
    size n. The optional keyword argument ny allows for a different
    size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='valid')

    return(improc)


def smooth1d(x, window_len=10, window='hanning'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    Parameters
    ----------
    x: array_like
        The input signal
    window_len: int
        The dimension of the smoothing window
    window: str, optional
        The type of window from 'flat', 'hanning', 'hamming', 'bartlett',
        'blackman'. Flat window will produce a moving average smoothing.

    Returns:
    --------
    x_smooth: ndarray
        The smoothed signal

    Examples:
    ---------
    >>> import numpy as np
    >>> from pyvm.models.tools import smooth1d
    >>> t = np.linspace(-2, 2, 10)
    >>> x = np.sin(t) + np.random.randn(len(t)) * 0.1
    >>> y = smooth1d(x)

    See also:
    ---------
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman,
    numpy.convolve
    scipy.signal.lfilter
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming',"
                         + "'bartlett', 'blackman'")

    s = np.r_[2 * x[0] - x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]

    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')

    return y[window_len-1:-window_len+1]


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

    def define_variable_layer_gradient(self, ilyr, dvdz, v0=None):
        """
        Replace velocities within a layer by defining a gtradient that
        varies linearly in the horizontal directions.

        Parameters
        ----------
        ilyr: int
            Index of layer to work on.
        dvdz: array_like
            List of velocity gradient values. Must have shape (nx, ny).
        v0: float, array_like or None, optional
            List of velocities at the top of the layer. Must be of
            shape (nx, ny), a scalar value, or None. Default is to use
            the value at the base of the overlying layer.
        """
        z0, z1 = self.get_layer_bounds(ilyr)

        if v0 is not None:
            v0 = np.atleast_1d(v0)
            if len(v0) == 1:
                v0 = v0 * np.ones((self.nx, self.ny))
            else:
                v0 = np.asarray(v0)

        assert (v0 is None) or (v0.shape == (self.nx, self.ny)),\
            'v0 must be scalar, nx-by-ny, or None'

        for ix in range(0, self.nx):
            for iy in range(0, self.ny):
                iz0, iz1 = self.grid.z2i((z0[ix, iy], z1[ix, iy]))
                iz1 += 1
                z = self.grid.z[iz0:iz1] - self.grid.z[iz0]
                if v0 is None:
                    if iz0 == 0:
                        _v0 = 0.
                    else:
                        _v0 = 1. / self.sl[ix, iy, iz0 - 1]
                else:
                    _v0 = v0[ix, iy]

                self.sl[ix, iy, iz0:iz1] = 1. / (_v0 + z * dvdz[ix, iy])

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

        Parameters
        ----------
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

    def smooth_interface(self, iref, n=1, nwin=3, nwin_y=None):
        """
        Smooth interface depth by convolving a gaussian kernal of typical size
        n.  The optional keyword argument ``ny`` allows for a different
        size in the y direction.

        Parameters
        ----------
        iref: int
            Index of interface to smooth.
        n: int
            Number of times to run the filter.
        nwin: int, optional
            Size of the guassian kernal. Default is 3.
        nwin_y: int, optional
            Specifies a different dimension for the smoothing
            kernal in the y direction. Default is to set the y size equal to
            to the size in x.
        """
        for i in range(n):
            nwin = min(self.nx, nwin)
            if nwin_y is None:
                nwin_y = nwin
            nwin_y = min(self.ny, nwin_y)
            if nwin_y == 1:
                self.rf[iref] = smooth1d(self.rf[iref].flatten(), nwin)\
                    .reshape((self.nx, 1))
            else:
                self.rf[iref] = smooth2d(self.rf[iref], nwin, ny=nwin_y)

    def fix_pinchouts(self, min_dz=None):
        """
        Fixes layer pinchouts so that boundaries do not cross.


        Parameters
        ----------
        min_dz: float
            Minimum layer thickness.  Default is the vertical grid spacing.
        """
        if min_dz is None:
            min_dz = self.dz

        # put boundaries in correct order
        for ir in range(1, self.nr):
            _top = self.rf[ir - 1]
            _bot = self.rf[ir]

            top = np.minimum(_top, _bot)
            bot = np.maximum(_top, _bot)

            self.rf[ir - 1] = top
            self.rf[ir] = bot

        # ensure minimum spacing between boundaries
        for ir in range(1, self.nr):
            dz = self.rf[ir] - self.rf[ir  - 1]
            idx = np.nonzero(dz < min_dz)
            self.rf[ir][idx] += min_dz  - dz[idx]
