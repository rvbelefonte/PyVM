"""
Exceptions and error checking VM Tomography models
"""
import warnings
import numpy as np

class VMError(Exception):
    pass

class VMGridError(VMError):
    """
    Raised if there is a problem with the slowness grid.
    """
    pass
    
class VMInterfaceError(VMError):
    """
    Raised if there is a problem with the model interfaces.
    """
    pass



class VMErrorChecking(object):
    """
    Convience class for error-checking methods
    """

    def verify(self, raise_error=False, warn=True):
        
        n = 0

        grid_errors = self._check_grid(raise_error=False, warn=False)
        if grid_errors:
            n += len(grid_errors)
            msg = ' {:} problems with slowness grid: '.format(len(grid_errors))
            msg += '; '.join(grid_errors)

        
        interface_errors = self._check_interfaces(raise_error=False, warn=False)
        if interface_errors:
            n += len(interface_errors)
            msg = ' {:} problems with interfaces: '.format(len(interface_errors))
            msg += '; '.join(interface_errors)


        if n == 0:
            return True


        msg = 'Found {:} total problems. '.format(n) + msg
        if raise_error:
            VMError(msg)
        elif warn:
            warnings.warn(msg)

        return False

    def _check_grid(self, raise_error=False, warn=True):
        """
        Check model grid for problems.
        """
        errors = []
        
        # make sure grid is a 3D numpy array
        if type(self.sl)!=np.ndarray:
            errors.append('`sl` is of type {:} (must be numpy.ndarray)'\
                          .format(type(self.sl)))
            
            return errors
            
        # check dimensions
        if len(self.sl.shape) != 3:
            errors.append('Grid must be 3D')
        
        # look for NaNs
        n = len(np.nonzero(np.isnan(self.sl))[0])
        if n > 0:
            errors.append('{:} NaN values'.format(n))
        
        # look for zeros in slowness
        n = len(np.nonzero(self.sl == 0.0)[0])
        if n > 0:
            errors.append('{:} zero slowness values'.format(n))
            
        if len(errors) == 0:
            return None

        msg = 'Found {:} problems with slowness grid: '.format(len(errors))
        msg += '; '.join(errors)
        
        if raise_error:
            raise VMGridError(msg)
        elif warn:
            warnings.warn(msg)
            
        return errors


    def _check_interfaces(self, raise_error=False, warn=True):
        """
        Check model interfaces for problems.
        """
        errors = []
        
        # make sure interfaces array is a 3D numpy array
        if type(self.rf) != np.ndarray:
            errors.append('`rf` array is of type {:} (must be numpy.ndarray)'\
                          .format(type(self.rf)))
            
            return errors
        
        # check dimensions
        if self.nr == 0:
            return None

        if self.rf.shape != (self.nr, self.nx, self.ny):
            errors.append('Mismatch between interface and grid shapes'\
                          + ' (rf.shape = {:} -- expected ({:}, {:}, {:}))'\
                          .format(self.rf.shape, self.nr, self.nx, self.ny))
            
        # check for NaNs
        n = len(np.nonzero(np.isnan(self.rf))[0])
        if n > 0:
            errors.append('{:} NaN values'.format(n))
            
        # look for infinite values
        n = len(np.nonzero(np.isinf(self.rf))[0])
        if n != 0:
            errors.append('{:} infinite values'.format(n))
            
        # check 
        
        if len(errors) == 0:
            return None

        msg = 'Found {:} problems with interfaces: '.format(len(errors))
        msg += '; '.join(errors)
        
        if raise_error:
            raise VMGridError(msg)
        elif warn:
            warnings.warn(msg)
            
        return errors
