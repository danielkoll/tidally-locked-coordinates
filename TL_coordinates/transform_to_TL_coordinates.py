import numpy as np
from scipy.interpolate import griddata

'''
* Tools for plotting GCM code in a tidally locked coordinate system.
* Original code by Daniel Koll.
* Improvements and testing by Huanzhou Yang.

For definitions and diagrams, see 
Appendix B in Koll & Abbot (2015).

--
This library transforms between Earth-like spherical 
coordinates (angles defined wrt. axis of rotation) and
tidally-locked spherical coordinates (angles defined wrt.
axis of insolation;
latitude = angle from terminator, substellar point = pi/2;
longitude = angle around substellar point.
--
NOTE:
To ensure that global averages are independent of the coordinates used, use nearest-neighbor interpolation.
Linear or higher-order schemes might not guarantee conservation properties.
'''


### CLASSES
class Dummy():
    pass



### FUNCTIONS
# NOTE:
#   these functions expect radians!
def transform_latlon_to_TL(lat_tmp,lon_tmp,lon_ss=0.):
    if lat_tmp.ndim==1 or lon_tmp.ndim==1:
        lon,lat = np.meshgrid(lon_tmp,lat_tmp)
    else:
        lat,lon = lat_tmp,lon_tmp
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    lon_ss = -lon_ss
    xprime = np.cos(lon_ss)*x - np.sin(lon_ss)*y
    yprime = np.sin(lon_ss)*x + np.cos(lon_ss)*y

    lat_TL = np.arcsin(xprime)
    lon_TL = np.arctan2(yprime,z)
    # change from lon in {-pi,pi} to lon in {0,2pi}:
    lon_TL[lon_TL<0.] = lon_TL[lon_TL<0.] + 2.*np.pi

    return lat_TL, lon_TL

def transform_velocities_to_TL(u,v,lat_tmp,lon_tmp,lon_ss=0.):
    # This returns TL velocities but on irregular grid.
    if lat_tmp.ndim==1 or lon_tmp.ndim==1:
        lon,lat = np.meshgrid(lon_tmp,lat_tmp)
    else:
        lat,lon = lat_tmp,lon_tmp
    lat_TL,lon_TL = transform_latlon_to_TL(lat,lon,lon_ss)

    lon_ss = -lon_ss

    # (tedious algebra - got these via Mathematica)
    Dlon_TL_Dlon = 1./( 1./np.cos(lon+lon_ss)*np.tan(lat)+np.cos(lat)/np.sin(lat)*np.sin(lon+lon_ss)*np.tan(lon+lon_ss) )
    Dlon_TL_Dlat = 8.*np.sin(lon+lon_ss) / ( -6.+2.*np.cos(2.*lat)+np.cos(2.*(lat-lon-lon_ss))+\
                                                  2.*np.cos(2*(lon+lon_ss))+np.cos(2.*(lat+lon+lon_ss)) )
    Dlat_TL_Dlon = -np.cos(lat)*np.sin(lon+lon_ss)/( np.sqrt(1.-np.cos(lat)**2*np.cos(lon+lon_ss)**2) )
    Dlat_TL_Dlat = -np.cos(lon+lon_ss)*np.sin(lat)/( np.sqrt(1.-np.cos(lat)**2*np.cos(lon+lon_ss)**2) )

    u_TL = Dlon_TL_Dlon * np.cos(lat_TL)/np.cos(lat)*u + Dlon_TL_Dlat*np.cos(lat_TL)*v
    v_TL = Dlat_TL_Dlon * u/np.cos(lat) + Dlat_TL_Dlat*v
    return u_TL,v_TL

def transform_velocities_to_TL_interp(u,v,lat,lon,lat_TL,lon_TL,lon_ss=0.,method="nearest"):
    # This returns TL velocities, interpolated onto regular grid.
    u_TL,v_TL = transform_velocities_to_TL(u,v,lat,lon,lon_ss)
    u_TL_i = interpolate_to_TL_ndim(lat,lon,lat_TL,lon_TL,u_TL,lon_ss,method=method)
    v_TL_i = interpolate_to_TL_ndim(lat,lon,lat_TL,lon_TL,v_TL,lon_ss,method=method)
    return u_TL_i,v_TL_i

def interpolate_to_TL(lat,lon,lat_TL,lon_TL,data,lon_ss=0.,method="nearest"):
    # Assume data is 2D!
    # Interpolate data, given on an earth-like lat-lon grid, to a TL lat-lon grid.
    # First, transform the given lat-lon points into TL coords.
    # Second, use the given points as basis for interpolation.
    lat_TL_given,lon_TL_given = transform_latlon_to_TL(lat,lon,lon_ss)
    data_interp = griddata( (lat_TL_given.ravel(),lon_TL_given.ravel()),\
                                data.ravel(),(lat_TL,lon_TL),method=method )
    return data_interp

def interpolate_to_TL_ndim(lat,lon,lat_TL,lon_TL,data,lon_ss=0.,method="nearest"):
    # Uses above, but for N-dim data.
    # Also assume lat/lon are last two dims.
    if data.ndim==2:
        data_interp = interpolate_to_TL(lat,lon,\
                                            lat_TL[None,:],\
                                            lon_TL[:,None],data,lon_ss,method).T
    elif data.ndim==3:
        data_interp = np.zeros((data.shape[0],lat_TL.size,lon_TL.size))
        for i in range(data.shape[0]):
            data_interp[i,...] = \
                interpolate_to_TL(lat,lon,lat_TL[None,:],lon_TL[:,None],\
                                      data[i,...],lon_ss,method).T
    elif data.ndim==4:
        data_interp = np.zeros((data.shape[0],data.shape[1], \
                                lat_TL.size,lon_TL.size))
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                data_interp[i,j,...] = \
                                     interpolate_to_TL(lat,lon,\
                                        lat_TL[None,:],lon_TL[:,None],\
                                        data[i,j,...],lon_ss,method).T
    else:
        # fix this later...x
        print "(interpolate_to_TL_ndim) Error! expected 4 or fewer dimensions"
        pass

    return data_interp



#### ------------------
# MAIN FUNCTION FOR EXTERNAL USERS HERE:
# This function expects a Earth-coord state object
# (= dummy with attached coords and data).
# Returns TL-coord state with TL data attached.
# Assume that lat/lon/lon_ss are all in degrees!
#
# CAUTION:
#   to ensure conservation of global means.
#    Nlat/Nlon for TL coords should match the GCM's lat/lon resolution in Earth-like coordinate!
#   only pick larger values for making smooth plots in which conservation doesn't matter.

def transform_state(state,vars_list,(Nlat,Nlon),lon_ss=0.,do_vel=False):
    degtorad = np.pi/180.
    state_TL = Dummy()    # placeholder object
    
    state_TL.lon_ss = lon_ss*degtorad
    state_TL.p = state.p
    #state_TL.phalf = state.phalf
    state_TL.t = state.t
    state_TL.lat = np.linspace(-90,90,Nlat)
    state_TL.lon = np.linspace(0,360,Nlon)
    state_TL.weights = np.cos(state_TL.lat*degtorad)

    method = "nearest"   # which interpolation method?

    for var in vars_list:
        ## skip if velocities in list, do that at end
        if var[0]=="U" or var=="ucomp":
            var_u = var
        elif var[0]=="V" or var=="vcomp":
            var_v = var
        else:
            x = getattr(state,var)
            xi = interpolate_to_TL_ndim(state.lat*degtorad,state.lon*degtorad,\
                                       state_TL.lat*degtorad,\
                                       state_TL.lon*degtorad,\
                                       x,lon_ss=state_TL.lon_ss,method=method)
            setattr(state_TL,var,xi)

    if do_vel:
        u = getattr(state,var_u)
        v = getattr(state,var_v)
        ui,vi = transform_velocities_to_TL_interp( \
                    u,v,state.lat*degtorad,state.lon*degtorad,\
                    state_TL.lat*degtorad,state_TL.lon*degtorad, \
                    lon_ss=state_TL.lon_ss,method=method)
        setattr(state_TL,var_u,ui)
        setattr(state_TL,var_v,vi)

    return state_TL
