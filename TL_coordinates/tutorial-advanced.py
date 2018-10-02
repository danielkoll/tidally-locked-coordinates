import numpy as np
import matplotlib.pyplot as plt


import get_GCM_data
import transform_to_TL_coordinates

### ---------------------------
# helpers, definitions



### ---------------------------
# get data, compute etc.

output = {'TS':'TS','CLDTOT':'CLDTOT','U':'U','V':'V'}
state = get_GCM_data.get_GCM("./", ["GCMOutput.nc"], vars_list=output)


# now convert to TL coords
#   here: pick Nlat/Nlon to match GCM's resolution in Earth-like coordinates.
#   otherwise can get different global means!

nlat_tl = state.lat.size  # what resolution desired for interp into TL coords?
nlon_tl = state.lon.size  # 
lon_ss = 180. # at which lon is the substellar point?
state_TL = transform_to_TL_coordinates.transform_state(state,output,(nlat_tl,nlon_tl),lon_ss=lon_ss,do_vel=True)

# note: in this case the time coord only has one entry.
#   get rid of singleton dims with np.squeeze()
for s in [state,state_TL]:
    for var in output:
        x0 = getattr(s,var)
        x1 = np.squeeze(x0)
        setattr(s,var,x1)

print "coordinate sizes: ",state.time.shape,state.lat.shape,state.lon.shape
print "output var sizes: ",state.TS.shape



# DONE!

# ---
## everything below this line is just diagnostics to help plot the exact coordinate points
degtorad = np.pi/180.
lat_in_tl,lon_in_tl = transform_to_TL_coordinates.transform_latlon_to_TL(state.lat*degtorad,state.lon*degtorad,lon_ss=lon_ss)
lat_in_tl = lat_in_tl/degtorad
lon_in_tl = lon_in_tl/degtorad

lat_2d,lon_2d = np.meshgrid(state.lat,state.lon)

## also check global means:
print "global mean Ts in Earth coords = %.3fK" % np.average(np.average(state.TS,axis=-1) * state.weights,axis=-1)
print "global mean Ts in TL coords = %.3fK" % np.nanmean(np.nanmean(state_TL.TS,axis=-1) * state_TL.weights,axis=-1)

### ---------------------------
# make plots

cmap = 'RdBu_r'

plt.figure(figsize=[8,3.])
# -
plt.subplot(1,2,1)
plt.pcolormesh(state.lon,state.lat,state.TS,cmap=cmap,vmin=200,vmax=300)
plt.colorbar()
plt.scatter(lon_2d,lat_2d,s=4,c="k")
plt.xlim(0,360)
plt.ylim(-90,90)
plt.title('TS,Earth coordinates')
# -
plt.subplot(1,2,2)
plt.pcolormesh(state_TL.lon,state_TL.lat,state_TL.TS,cmap=cmap,vmin=200,vmax=300)
plt.colorbar()
plt.scatter(lon_in_tl,lat_in_tl,s=4,c="k")
plt.xlim(0,360)
plt.ylim(-90,90)
plt.title('TS,TL coordinates')



plt.figure(figsize=[8,3.])
# -
plt.subplot(1,2,1)
plt.pcolormesh(state.lon,state.lat,state.CLDTOT,cmap=cmap,vmin=0,vmax=1)
plt.colorbar()
plt.xlim(0,360)
plt.ylim(-90,90)
plt.title('CLDTOT,Earth coordinates')
# -
plt.subplot(1,2,2)
plt.pcolormesh(state_TL.lon,state_TL.lat,state_TL.CLDTOT,cmap=cmap,vmin=0,vmax=1)
plt.colorbar()
plt.xlim(0,360)
plt.ylim(-90,90)
plt.title('CLDTOT,TL coordinates')


#
ind_p = -1   # desired index in vertical, e.g., mid-troposphere or near surface
n = 5   # how dense to plot wind vectors?

plt.figure(figsize=[8,3.])
# -
plt.subplot(1,2,1)
CS = plt.pcolormesh(state.lon,state.lat,state.TS,cmap=cmap,vmin=200,vmax=300)
Q = plt.quiver(state.lon[::n],state.lat[::n],state.U[ind_p,::n,::n],state.V[ind_p,::n,::n],pivot='mid')
qk = plt.quiverkey(Q, 0.45, 0.95, 10, r'$10 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.colorbar(CS)
plt.xlim(0,360)
plt.ylim(-90,90)
plt.title('TS+wind,Earth coordinates')
# -
plt.subplot(1,2,2)
CS = plt.pcolormesh(state_TL.lon,state_TL.lat,state_TL.TS,cmap=cmap,vmin=200,vmax=300)
Q = plt.quiver(state_TL.lon[::n],state_TL.lat[::n],state_TL.U[ind_p,::n,::n],state_TL.V[ind_p,::n,::n],pivot='mid')
qk = plt.quiverkey(Q, 0.9, 0.95, 10, r'$10 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.colorbar(CS)
plt.xlim(0,360)
plt.ylim(-90,90)
plt.title('TS+wind,TL coordinates')


plt.show()
