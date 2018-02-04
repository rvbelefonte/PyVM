"""
Input/output tools for VM models
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import os
import numpy as np
from struct import unpack
from pyvm.io import pack, unpack
from pyvm.utils.loaders import get_example_file

ENDIAN = pack.BYTEORDER

class VMIO(object):
    """
    Convience class for VM model I/O
    """
    def read(self, path_or_buf, fmt=None, **kwargs):
        """
        Read a VM model from a file

        Parameters
        ----------
        path_or_buf: string or file handle
            File path or object to read model from.
        fmt: string, optional
            Format of file to read. Default is to try to guess the format
            from the file extension or try to read in the native VM (*.vm)
            format. Supported formats must be accessible by _read_{fmt}
            private methods.
        kwargs:
            Keyword arguments that are passed to the read methods. See
            _read_{fmt} methods for advanced options.
        """
        if not hasattr(path_or_buf, 'read'):
            try:
                buf = open(path_or_buf, 'rb')
                close = True
            except IOError as err:
                try:
                    buf = open(get_example_file(path_or_buf), 'rb')
                    close = True
                except:
                    raise IOError(err)
        else:
            buf = path_or_buf

        if (fmt is None) and (not hasattr(path_or_buf, 'read')):
            try:
                _, fmt = os.path.splitext(path_or_buf)
                fmt = fmt[1:]
                method = self.__getattribute__('_read_{:}'.format(fmt))
            except AttributeError:
                fmt = 'vm'
                method = self.__getattribute__('_read_{:}'.format('vm'))
        else:
            method = self.__getattribute__('_read_{:}'.format(fmt))

        method(buf, **kwargs)
        
        if close:
            buf.close()

    def _read_vm(self, buf, head_only=False, endian=ENDIAN):

        # Read header
        nx, ny, nz, nr  = unpack.unpack_4byte_Integer(buf, 4, endian=endian)
        if nz == 0:
            nz = ny
            ny = 1

        origin = unpack.unpack_4byte_IEEE(buf, 3, endian=endian)
        _r2 = unpack.unpack_4byte_IEEE(buf, 3, endian=endian)
        spacing = unpack.unpack_4byte_IEEE(buf, 3, endian=endian)

        # initialize the model
        self._init_model((nx, ny, nz), origin=origin, spacing=spacing)

        if head_only:
            return

        # Slowness grid
        ngrid = nx * ny * nz
        sl = np.asarray(unpack.unpack_4byte_IEEE(buf, ngrid, endian=endian))
        self.grid.values = np.reshape(sl, (self.grid.shape))

        # Interface depths and slowness jumps
        nintf = nx * ny * nr
        self.rf = np.reshape(
            unpack.unpack_4byte_IEEE(buf, nintf, endian=endian), (nr, nx, ny))
        self.jp = np.reshape(
            unpack.unpack_4byte_IEEE(buf, nintf, endian=endian), (nr, nx, ny))

        # Interface flags
        self.ir = np.reshape(
            unpack.unpack_4byte_Integer(buf, nintf, endian=endian),
            (nr, nx, ny))
        self.ir -= 1

        self.ij = np.reshape(
            unpack.unpack_4byte_Integer(buf, nintf, endian=endian),
            (nr, nx, ny))
        self.ij -= 1

    def write(self, path_or_buf, fmt=None, **kwargs):
        """
        Write VM model to a file

        Parameters
        ----------
        path_or_buf: string or file handle
            File path or object to write model to.
        fmt: string, optional
            Format to write file to. Default is to try to guess the format
            from the filename extension or write in the native VM (*.vm)
            format. Supported formats must be accessible by _write_{fmt}
            private methods.
        kwargs:
            Keyword arguments that are passed to the read methods. See
            _write_{fmt} methods for advanced options.
        """
        if not hasattr(path_or_buf, 'write'):
            buf = open(path_or_buf, 'wb')
            close = True
        else:
            buf = path_or_buf

        if (fmt is None) and (not hasattr(path_or_buf, 'write')):
            try:
                _, fmt = os.path.splitext(path_or_buf)
                fmt = fmt[1:]
                method = self.__getattribute__('_write_{:}'.format(fmt))
            except AttributeError:
                fmt = 'vm'
                method = self.__getattribute__('_write_{:}'.format('vm'))
        else:
            method = self.__getattribute__('_write_{:}'.format(fmt))

        method(buf, **kwargs)

        if close:
            buf.close()

    def _write_vm(self, buf, endian=ENDIAN):

        for v in [self.nx, self.ny, self.nz, self.nr]:
            pack.pack_4byte_Integer(buf, np.int32(v), endian)

        for r in (self.r1, self.r2):
            for v in r:
                pack.pack_4byte_IEEE(buf, np.float32(v), endian)
        
        for v in [self.dx, self.dy, self.dz]:
            pack.pack_4byte_IEEE(buf, np.float32(v), endian)

        sl = np.reshape(self.sl, (self.nx * self.ny * self.nz))
        pack.pack_4byte_IEEE(buf, np.float32(sl), endian)

        if self.nr > 0:
            nrefl = self.nr * self.nx * self.ny
            for attr in ['rf', 'jp']:
                v = np.reshape(self.__getattribute__(attr), (nrefl))
                pack.pack_4byte_IEEE(buf, np.float32(v), endian)
        
            for attr in ['ir', 'ij']:
                v = np.reshape(self.__getattribute__(attr), (nrefl)) + 1
                pack.pack_4byte_Integer(buf, np.int32(v), endian)

    def _write_bin(self, buf, order=(0, 1, 2), dtype=np.float32):

        self.grid.to_bin(buf, order=order, dtype=dtype)
