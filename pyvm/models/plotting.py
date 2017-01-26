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
            aspect='auto', velocity=True, apply_jumps=True,
            colorbar=False,
            vmin=None, vmax=None, rf=True, ir=True, ij=True, show=None):

        #XXX TODO
        assert self.ny == 1, 'Only works for 2D models, for now'
        vm = self.copy() #TODO placeholder for 2D slice from 3D model

        if apply_jumps:
            vm.apply_jumps()

        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            new = True
        else:
            new = False

        # grid
        img = np.flipud(vm.sl[:, 0, :].T)
        if velocity:
            img = 1. / img
        c = ax.imshow(img, extent=[vm.r1[0], vm.r2[0], vm.r1[2], vm.r2[2]],
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

        if colorbar:
            plt.colorbar(c)

        if new:
            plt.xlim(vm.r1[0], vm.r2[0])
            plt.ylim(vm.r1[2], vm.r2[2])
            plt.gca().invert_yaxis()

        if ((show is None) and new) or show:
            plt.show()
