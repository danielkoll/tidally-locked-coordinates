##import Nio  #Need PyNio for reading netCDF files  ## DKOLL: outdated?
from scipy.io import netcdf                         ## DKOLL: this should work
import numpy

### FUNCTIONS
# This module contains functions that read NetCDF data from different GCMs
class Dummy():
    pass

class Variable():
    # Might be able to do this with a dict instead, but this way
    # I can include potential conversion factors to deal with non-standard units...
    # use 'fac' to account for desired unit conversions, e.g., mbar->Pa
    def __init__(self,abbrev,name,fac=1.):
        self.abbrev = abbrev
        self.name = name
        self.factor = fac

## Get GCM output from a netcdf, and return as an object.
# The default coordinate names need to be adapted based on GCM!
# Below are variables that work for CESM.
#
# Input: list of files and variable names.
#
# Output: a single object that has the GCM's variables as well as output attached.
#
# Options:
#    zonalonly = have this function only return zonally averaged data?
# Notes:
#    'filenames' needs to be a list!
#
# Example: state = get_GCM("./", ["output01.nc"], vars_list={'TS':'TS','CLDTOT':'CLDTOT'})

def get_GCM(path, filenames, zonalonly=False, vars_list={}):

    state = Dummy()

    # DEFINE VARIABLES (this can vary according to model!):
    AllVars = [ Variable("t","time"), \
                Variable("lat","lat"), \
                Variable("lon","lon"), \
                Variable("p","lev")]
    for varkey in vars_list:
        AllVars.append( Variable(varkey,vars_list[varkey]) )

    state.time = numpy.array([0.])

    # GET VARIABLES & STORE THEM:
    for fname in filenames:
        ##f = Nio.open_file(path+fname)
        f = netcdf.netcdf_file(path+fname,"r")

        for var in AllVars:
            # get the data, also deal with 0-d variables (e.g. P0)
            # if len( f.variables[var.name] ) > 0:
            if list( f.variables[var.name] ) > 0:
                x = f.variables[var.name][:] * var.factor
            else:
                x = f.variables[var.name].get_value() * var.factor
                
            if x.ndim > 2:       # average over last dim (usually, longitude)
                if zonalonly:
                    x = numpy.average(x,axis=x.ndim-1)
                else:
                    pass
                    
                # if state already contains the variable,
                # append along time (first) dimension
                old_x = getattr(state, var.abbrev, [])
                if len(old_x) > 0:
                    setattr(state, var.abbrev, numpy.concatenate( (old_x,x),axis=0))
                else:
                    setattr(state, var.abbrev, x)
            else:
                setattr(state, var.abbrev, x)

        state.weights = numpy.cos(state.lat*numpy.pi/180.) #weights for meridional averaging

        # For time dim: if every file resets time, keep adding it up
        state.time = numpy.concatenate( (state.time,state.t)) #+max(state.time)) )
        f.close()

    state.time = state.time[1:] # need to remove spurious first element again
    return state

