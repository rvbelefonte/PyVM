"""
Main module for managing tomography
"""
from pyvm.models.vm import VM
from pyvm.picks.pickdb import PickDatabase

class VMTomographer(object):

    def __init__(self, pickdb=None, model=None, rays=None):

        if pickdb:
            self.pickdb = pickdb
        else:
            self.pickdb = PickDatabase()

        if isinstance(model, VM):
            # connect to existing model instance
            self.model = model
        else:
            # create a new default model or read existing file
            self.model = VM(filename=model)

        if rays:
            self.rays = rays

    #XXX function prototypes
    def raytrace(self):
        # TODO
        # should raytrace and store output in self.rays
        raise NotImplementedError
    
    def invert(self):
        # TODO
        # should update self.model using rays in self.rays
        raise NotImplementedError

    def plot(self, xlim=None, ylim=[0.0, 0.0], zlim=None, model=True,
            rays=True, picks=True):
        # TODO
        # plot data along a slice
        raise NotImplementedError
