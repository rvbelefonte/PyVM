"""
Tools for plotting VM models
"""
from __future__ import (absolute_import, division, print_function,
        unicode_literals)

import numpy as np
import matplotlib.pyplot as plt


class VMPlotter(object):
    """
    Convience class for VM model plotting tools 
    """
    def plot(self, ax=None, figsize=None, cmap='jet',
            aspect=None, velocity=True, vmin=0.333, vmax=8.5,
            rf=True, ir=True, ij=True):

        #XXX TODO
        assert self.ny == 1, 'Only works for 2D models, for now'
        vm = self #TODO placeholder for 2D slice from 3D model

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            show = True

        # grid
        img = np.flipud(vm.sl[:, 0, :].T)
        if velocity:
            img = 1. / img
        ax.imshow(img, extent=[vm.r1[0], vm.r2[0], vm.r1[2], vm.r2[2]],
                cmap=cmap, vmin=vmin, vmax=vmax, aspect=aspect)

        # boundaries
        for iref in range(vm.nr):

            if rf:
                ax.plot(vm.grid.x, vm.rf[iref, :, 0], '-w', lw=2)
            if ir:
                _ir = vm.rf[iref, :, 0]
                idx = np.nonzero(_ir == 0)
                _ir[idx] = np.nan
                ax.plot(vm.grid.x, _ir, '-k', lw=1)
            if ij:
                _ij = vm.rf[iref, :, 0]
                idx = np.nonzero(_ij == 0)
                _ij[idx] = np.nan
                ax.plot(vm.grid.x, _ij, '-g', lw=0.5)

        plt.xlim(vm.r1[0], vm.r2[0])
        plt.ylim(vm.r1[2], vm.r2[2])
        if show:
            plt.gca().invert_yaxis()
            plt.show()
