"""
Support for working with VM Tomography binary rayfan files.
"""
import os
import warnings
from struct import unpack
from pyvm.io import pack
import logging
import numpy as np

ENDIAN = pack.BYTEORDER
DEFAULT_RAYFAN_VERSION = 2


class RayfanError(Exception):
    """
    Base exception class for the Rayfan class.
    """
    pass


class RayfanReadingError(RayfanError):
    """
    Raised if there is a problem reading a rayfan from the disk.
    """
    pass


class RayfanNotFoundError(RayfanError):
    """
    Raised if a rayfan is not found in the ray file.
    """
    pass


class RayfanGroup(object):
    """
    Class for working with VM Tomography rayfan files.
    """
    def __init__(self, endian='@'):
        """
        Class for working with VM Tomography rayfan files.

        :param file: An open file-like object or a string which is
            assumed to be a filename.
        :param endian: Optional. The endianness of the file.
            Default is to use machine's native byte order.
        """
        self.file = None
        self.rayfans = []

    def __str__(self):
        """
        Print a summary of the rayfan.
        """
        sng = '{:}:'.format(os.path.basename(self.file.name))
        sng += ' nrayfans = {:},'.format(len(self.rayfans))
        sng += ' Chi^2 = {:}, rms = {:}'.format(self.chi2, self.rms)
        return sng

    def read(self, file, endian='@'):
        """
        Read rayfan data from a rayfan file.

        :param file: An open file-like object or a string which is
            assumed to be a filename.
        :param endian: Optional. The endianness of the file. Default is
            to use machine's native byte order.
        """
        if not hasattr(file, 'read') or not hasattr(file, 'tell') or not \
            hasattr(file, 'seek'):
            file = open(file, 'rb')
        else:
            file.seek(0)
        self.file = file
        self._read(file, endian=endian)

    def _read(self, file, endian='@'):
        """
        Read rayfan data from a rayfan file.

        :param file: An open file-like object with the file pointer set at the
            beginning of a rayfan file.
        :param endian: Optional. The endianness of the file. Default is
            to use machine's native byte order.
        """
        # Read file header
        fmt = '{:}i'.format(endian)
        n = unpack(fmt, file.read(4))[0]
        if n < 0:
            self.FORMAT = -n
            n = unpack(fmt, file.read(4))[0]
        else:
            self.FORMAT = 1
        # Read the individual rayfans
        self.rayfans = []
        for i in range(0, n):
            self.rayfans.append(Rayfan(file, endian=endian,
                                       version=self.FORMAT))

    def write(self, filename, version=DEFAULT_RAYFAN_VERSION, endian=ENDIAN):
        """
        Write a rayfan group

        ..warning: Overwrites existing filename.

        Parameters
        ----------
        filename: str
            Filename to write rayfan to.
        version: int
            Sets the version number of the rayfan file format. Default is version 2.
        endian: str
            Sets endianness of the file. Default is to use machine's native byte order.
        """
        file = open(filename, 'wb')

        # write version flag (only included for version >1)
        if version > 1:
            pack.pack_4byte_Integer(file, np.int32(-version), endian=endian)

        # number of rayfans in the group file
        pack.pack_4byte_Integer(file, np.int32(len(self.rayfans)), endian=endian)

        # write each rayfan
        for rfn in self.rayfans:
            rfn.write(file, version=version, endian=endian)


    def _get_all_azimuths(self):
        """
        Returns a list of all azimuths in the rayfans.
        """
        return np.concatenate([rfn.azimuths for rfn in self.rayfans])
    azimuths = property(fget=_get_all_azimuths)

    def _get_all_offsets(self):
        """
        Returns a list of all azimuths in the rayfans.
        """
        return np.concatenate([rfn.offsets for rfn in self.rayfans])
    offsets = property(fget=_get_all_offsets)

    def _get_all_residuals(self):
        """
        Returns a list of all residuals in the rayfans.
        """
        return np.concatenate([rfn.residuals for rfn in self.rayfans])
    residuals = property(fget=_get_all_residuals)

    def _get_all_bottom_points(self):
        """
        Returns a list of all ray bottom points.
        """
        return np.concatenate([rfn.bottom_points for rfn in self.rayfans])
    bottom_points = property(fget=_get_all_bottom_points)

    def _get_nrays(self):
        """
        Returns the total number of rays in all rayfans.
        """
        return len(self._get_all_offsets())
    nrays = property(fget=_get_nrays)

    def _calc_mean_rms(self):
        """
        Calculate mean RMS of all rayfans.
        """
        return np.mean([rfn.rms for rfn in self.rayfans])
    rms = property(fget=_calc_mean_rms)

    def _calc_mean_chi2(self):
        """
        Calculate mean Chi-squared value for all rayfans.
        """
        return np.mean([rfn.chi2_mean for rfn in self.rayfans])
    chi2 = property(fget=_calc_mean_chi2)


class Rayfan(object):
    """
    Class for working with a single rayfan.
    """
    def __init__(self, file, endian='@', version=DEFAULT_RAYFAN_VERSION):
        """
        Class for handling an individual rayfan.

        :param file: An open file-like object with the file pointer set at the
            beginning of a rayfan file.
        :param endian: Optional. The endianness of the file. Default is
            to use machine's native byte order.
        :param version: Optional. Sets the version number of the
            rayfan file format. Default is version 2.
        """
        self.read(file, endian=endian, version=version)

    def read(self, file, endian='@', version=DEFAULT_RAYFAN_VERSION):
        """
        Read data for a single rayfan.

        :param file: An open file-like object with the file pointer set at the
            beginning of a rayfan file.
        :param endian: Optional. The endianness of the file. Default is
            to use machine's native byte order.
        :param version: Optional. Sets the version number of the
            rayfan file format. Default is version 2.
        """
        filesize = os.fstat(file.fileno()).st_size
        # read the rayfan header information
        fmt = '{:}i'.format(endian)
        self.start_point_id = unpack(fmt, file.read(4))[0]
        self.nrays = unpack(fmt, file.read(4))[0]
        nsize = unpack(fmt, file.read(4))[0]
        # make sure the expected amount of data exist
        pos = file.tell()
        data_left = filesize - pos
        data_needed = 7 * self.nrays + 3 * nsize
        if data_needed > data_left or data_needed < 0:
            msg = '''
                  Too little data in the file left to unpack. This is most
                  likely caused by an incorrect size in the rayfan header.
                  '''.strip()
            raise RayfanReadingError(msg)
        # Static correction
        if version > 1:
            fmt = '{:}f'.format(endian)
            self.static_correction = unpack(fmt, file.read(4))[0]
        else:
            self.static_correction = 0.
        # ID arrays and sizes
        fmt = '{:}'.format(endian) + 'i' * self.nrays
        self.end_point_ids = unpack(fmt, file.read(4 * self.nrays))
        self.event_ids = unpack(fmt, file.read(4 * self.nrays))
        self.event_subids = unpack(fmt, file.read(4 * self.nrays))
        lens = unpack(fmt, file.read(4 * self.nrays))  # raypath length
        # Picks, travel-times, and errors
        fmt = '{:}'.format(endian) + 'f' * self.nrays
        self.pick_times = unpack(fmt, file.read(4 * self.nrays))
        self.travel_times = unpack(fmt, file.read(4 * self.nrays))
        self.pick_errors = unpack(fmt, file.read(4 * self.nrays))
        # Actual ray path coordinates
        self.paths = []
        for i in range(0, self.nrays):
            fmt = '{:}'.format(endian) + 'f' * 3 * lens[i]
            self.paths.append(np.reshape(
                unpack(fmt, file.read(4 * 3 * lens[i])),
                (lens[i], 3)))
        # Endpoint coordinates
        self.endpoints = []
        for path in self.paths:
            if len(path) > 0:
                self.endpoints.append(path[0])
            else:
                self.endpoints.append([None, None, None])

    def write(self, file, endian=ENDIAN, version=DEFAULT_RAYFAN_VERSION):
        """
        Write data for a single rayfan
        """
        # rayfan header information
        pack.pack_4byte_Integer(file, np.int32(self.start_point_id,
                                               endian=endian))
        pack.pack_4byte_Integer(file, np.int32(self.nrays), endian=endian)
        nsize = sum([len(p) for p in self.paths])
        pack.pack_4byte_Integer(file, np.int32(nsize),
                                endian=endian)

        if version > 1:
            pack.pack_4byte_IEEE(file, np.float32(self.static_correction), endian=endian)

        # ray id arrays and sizes
        pack.pack_4byte_Integer(file, np.asarray(self.end_point_ids,
                                                 dtype=np.int32), 
                                endian=endian)
        pack.pack_4byte_Integer(file, np.asarray(self.event_ids,
                                                 dtype=np.int32),
                                endian=endian)
        pack.pack_4byte_Integer(file, np.asarray(self.event_subids,
                                                 dtype=np.int32),
                                endian=endian)
        pack.pack_4byte_Integer(file, np.asarray([len(p) for p in self.paths],
                                                 dtype=np.int32),
                                endian=endian)

        # picks, travel-times, and errors
        pack.pack_4byte_IEEE(file, np.asarray(self.pick_times,
                                              dtype=np.float32),
                             endian=endian)
        pack.pack_4byte_IEEE(file, np.asarray(self.travel_times,
                                              dtype=np.float32),
                             endian=endian)
        pack.pack_4byte_IEEE(file, np.asarray(self.pick_errors,
                                              dtype=np.float32),
                             endian=endian)

        # ray path coordinates
        for p in self.paths:
            _path = np.asarray(p, dtype=np.float32).flatten()
            pack.pack_4byte_IEEE(file, _path, endian=endian)
        
    def _calc_offsets(self):
        """
        Calculate source-reciever offset for each path.
        """
        x = np.zeros(len(self.paths))
        for i, path in enumerate(self.paths):
            deltx = path[-1][0] - path[0][0]
            delty = path[-1][1] - path[0][1]
            x[i] = np.sqrt(deltx ** 2 + delty ** 2)
        return x
    offsets = property(fget=_calc_offsets)

    def _calc_azimuths(self):
        """
        Calculate source-receiver azimuth for each path.
        """
        az = np.zeros(len(self.paths))
        for i, path in enumerate(self.paths):
            deltx = path[-1][0] - path[0][0]
            delty = path[-1][1] - path[0][1]
            az[i] = (90 - np.rad2deg(np.arctan2(delty, deltx)))
            if az[i] < 0:
                az[i] += 360.
        return az
    azimuths = property(fget=_calc_azimuths)

    def _calc_residuals(self):
        """
        Calculate residuals.
        """
        return np.asarray(self.travel_times) - np.asarray(self.pick_times) \
            + self.static_correction
    residuals = property(fget=_calc_residuals)

    def _calc_rms(self):
        """
        Calculate the RMS of travel-time residuals.
        """
        return np.sqrt(np.sum(self.residuals ** 2) / self.nrays)
    rms = property(fget=_calc_rms)

    def _calc_chi2(self):
        """
        Calculate the Chi-squared value.
        """
        return (self.residuals / self.pick_errors) ** 2
    chi2 = property(fget=_calc_chi2)

    def _calc_mean_chi2(self):
        """
        Calculate the mean Chi-squared value.
        """
        return np.sum(self.chi2) / self.nrays
    chi2_mean = property(fget=_calc_mean_chi2)

    def _get_ray_bottom_points(self):
        """
        Find the bottoming point for each raypath.
        """
        pts = np.zeros((self.nrays, 3))
        for i, path in enumerate(self.paths):
            pts[i] = path[np.argmax([p[2] for p in path])]
        return pts
    bottom_points = property(fget=_get_ray_bottom_points)


def readRayfanGroup(file, endian=ENDIAN):
    """
    Read a VM tomography rayfan file.

    :param file: An open file-like object or a string which is
        assumed to be a filename.
    :param endian: Optional. The endianness of the file. Default is
        to use machine's native byte order.
    """
    rfn = RayfanGroup()
    rfn.read(file, endian=endian)
    return rfn


