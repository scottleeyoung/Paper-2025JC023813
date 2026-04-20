import xarray as xr
import numpy as np
import netCDF4 as nc
from datetime import datetime

import sys
sys.path.insert(0,"../modules")
import fortsa

def datetime_to_ordinal(dt):
    '''
    This function converts a datetime object to an ordinal object while considering the time of day.
    '''
    start_of_year = dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
    days_elapsed = (dt - start_of_year).days
    seconds_elapsed = dt.hour * 3600. + dt.minute * 60. + dt.second + dt.microsecond / 1e6
    ordinal_date = start_of_year.toordinal() + days_elapsed + seconds_elapsed / 86400.
    return ordinal_date

def str_datetime_2_ord(str_dt, ref_dt, fmt):
    '''
    This function converts date-time strings to ordinals.
    '''
    N = str_dt.size
    ref_ord = datetime_to_ordinal(datetime.strptime(ref_dt, fmt) )
    ordinal = np.zeros(N, dtype=np.float64)
    for i in range(N):
        dt_obj = datetime.strptime(str_dt[i], fmt)
        ordinal[i] = datetime_to_ordinal(dt_obj) - ref_ord
    return ordinal

def stat_sig(array1, array2, dt, max_time, sig_lvl=0.05, N=100):
   rng = np.random.default_rng(42)
   corr = np.empty(N)
   lag = int(max_time/dt)
   for i in range(N):
      x = rng.permutation(np.copy(array1))
      y = rng.permutation(np.copy(array2))
      corr[i] = np.max(fortsa.correlation(x, y, lag))
   return np.quantile(corr, 1-sig_lvl)

print("Reading detrended time series")
dt_est = 15/1440

ds = xr.open_dataset("data/detrended_driver_timeseries.nc", decode_times=False)
time = ds['time'].values
flow = ds['flow'].values
airt = ds['airt'].values
gs_salt = ds['gs_salt'].values
gs_temp = ds['gs_temp'].values
gs_tran = ds['gs_tran'].values
wind_u = ds['wind_u'].values
wind_v = ds['wind_v'].values
ds.close()

ds = xr.open_dataset("data/detrended_estuary_timeseries.nc")
zeta = ds['zeta'].values
salt = ds['salt'].values
temp = ds['temp'].values
ds.close()

ds = xr.open_dataset("data/HR1_salt-temp_detrended.nc", decode_times=False)
hr1_time = ds['time'].values
hr1_salt = ds['salt'].values
hr1_temp = ds['temp'].values
ds.close()

print("Computing cross-correlations")
indx = np.where(np.round(time,4) <= hr1_time[0])[0][-1]
NN = time[indx:].size

iter = 100
r95 = []
# zeta
Nlag = int(90/dt_est)
cor_tran = np.empty([Nlag*2+1,2])
cor_flow = np.empty([Nlag*2+1,2])
cor_windu = np.empty([Nlag*2+1,2])
cor_windv = np.empty([Nlag*2+1,2])
for i in range(2):
    cor_tran[:,i] = fortsa.correlation(gs_tran, zeta[:,i], Nlag)
    cor_flow[:,i] = fortsa.correlation(flow, zeta[:,i], Nlag)
    cor_windu[:,i] = fortsa.correlation(wind_u, zeta[:,i], Nlag)
    cor_windv[:,i] = fortsa.correlation(wind_v, zeta[:,i], Nlag)

    r95.append(stat_sig(gs_tran, zeta[:,i], dt_est, 90, sig_lvl=0.05, N=iter))
    r95.append(stat_sig(flow, zeta[:,i], dt_est, 90, sig_lvl=0.05, N=iter))
    r95.append(stat_sig(wind_u, zeta[:,i], dt_est, 90, sig_lvl=0.05, N=iter))
    r95.append(stat_sig(wind_v, zeta[:,i], dt_est, 90, sig_lvl=0.05, N=iter))

# salt
cor_gs_salt = np.empty([Nlag*2+1,3,2])
cor_flow_salt = np.empty([Nlag*2+1,3,2])
cor_gs_salt_hr1 = np.empty([Nlag*2+1,2])
cor_flow_salt_hr1 = np.empty([Nlag*2+1,2])
for i in range(3):
    for j in range(2):
        cor_gs_salt[:,i,j] = fortsa.correlation(gs_salt, salt[:,i,j], Nlag)
        cor_flow_salt[:,i,j] = fortsa.correlation(flow, salt[:,i,j], Nlag)

        r95.append(stat_sig(gs_salt, salt[:,i,j], dt_est, 90, sig_lvl=0.05, N=iter))
        r95.append(stat_sig(flow, salt[:,i,j], dt_est, 90, sig_lvl=0.05, N=iter))

        if i == 0:
            cor_gs_salt_hr1[:,j] = fortsa.correlation(gs_salt[indx:], hr1_salt[:NN,j], Nlag)
            cor_flow_salt_hr1[:,j] = fortsa.correlation(flow[indx:], hr1_salt[:NN,j], Nlag)

            r95.append(stat_sig(gs_salt[indx:], hr1_salt[:NN,j], dt_est, 90, sig_lvl=0.05, N=iter))
            r95.append(stat_sig(flow[indx:], hr1_salt[:NN,j], dt_est, 90, sig_lvl=0.05, N=iter))
            

# temp
cor_gs_temp = np.empty([Nlag*2+1,3,2])
cor_flow_temp = np.empty([Nlag*2+1,3,2])
cor_airt = np.empty([Nlag*2+1,3,2])
cor_gs_temp_hr1 = np.empty([Nlag*2+1,2])
cor_flow_temp_hr1 = np.empty([Nlag*2+1,2])
cor_airt_hr1 = np.empty([Nlag*2+1,2])
for i in range(3):
    for j in range(2):
        cor_gs_temp[:,i,j] = fortsa.correlation(gs_temp, temp[:,i,j], Nlag)
        cor_flow_temp[:,i,j] = fortsa.correlation(flow, temp[:,i,j], Nlag)
        cor_airt[:,i,j] = fortsa.correlation(airt, temp[:,i,j], Nlag)

        r95.append(stat_sig(gs_temp, temp[:,i,j], dt_est, 90, sig_lvl=0.05, N=iter))
        r95.append(stat_sig(flow, temp[:,i,j], dt_est, 90, sig_lvl=0.05, N=iter))
        r95.append(stat_sig(airt, temp[:,i,j], dt_est, 90, sig_lvl=0.05, N=iter))

        if i == 0:
            cor_gs_temp_hr1[:,j] = fortsa.correlation(gs_temp[indx:], hr1_temp[:NN,j], Nlag)
            cor_flow_temp_hr1[:,j] = fortsa.correlation(flow[indx:], hr1_temp[:NN,j], Nlag)
            cor_airt_hr1[:,j] = fortsa.correlation(airt[indx:], hr1_temp[:NN,j], Nlag)

            r95.append(stat_sig(gs_temp[indx:], hr1_temp[:NN,j], dt_est, 90, sig_lvl=0.05, N=iter))
            r95.append(stat_sig(flow[indx:], hr1_temp[:NN,j], dt_est, 90, sig_lvl=0.05, N=iter))
            r95.append(stat_sig(airt[indx:], hr1_temp[:NN,j], dt_est, 90, sig_lvl=0.05, N=iter))

print("Write results to NetCDF file")
lags = np.arange(-Nlag, Nlag+1) * dt_est

ds_out = nc.Dataset("data/lagged_xcorrelations_hr1.nc", mode='w', format='NETCDF4')
ds_out.createDimension("lags", lags.size)
ds_out.createDimension("two", 2)
ds_out.createDimension("three", 3)

ds_lags = ds_out.createVariable("lags", "f8", ("lags",))

ds_tran_zeta = ds_out.createVariable("cor_tran_zeta", "f8", ("lags","two",))
ds_flow_zeta = ds_out.createVariable("cor_flow_zeta", "f8", ("lags","two",))
ds_windu_zeta = ds_out.createVariable("cor_windu_zeta", "f8", ("lags","two",))
ds_windv_zeta = ds_out.createVariable("cor_windv_zeta", "f8", ("lags","two",))

ds_salt_salt = ds_out.createVariable("cor_salt_salt", "f8", ("lags","three","two",))
ds_flow_salt = ds_out.createVariable("cor_flow_salt", "f8", ("lags","three","two",))
ds_salt_salt_hr1 = ds_out.createVariable("cor_salt_salt_hr1", "f8", ("lags","two",))
ds_flow_salt_hr1 = ds_out.createVariable("cor_flow_salt_hr1", "f8", ("lags","two",))

ds_temp_temp = ds_out.createVariable("cor_temp_temp", "f8", ("lags","three","two",))
ds_flow_temp = ds_out.createVariable("cor_flow_temp", "f8", ("lags","three","two",))
ds_airt_temp = ds_out.createVariable("cor_airt_temp", "f8", ("lags","three","two",))
ds_temp_temp_hr1 = ds_out.createVariable("cor_temp_temp_hr1", "f8", ("lags","two",))
ds_flow_temp_hr1 = ds_out.createVariable("cor_flow_temp_hr1", "f8", ("lags","two",))
ds_airt_temp_hr1 = ds_out.createVariable("cor_airt_temp_hr1", "f8", ("lags","two",))

ds_lags[:] = lags
ds_tran_zeta[:] = cor_tran
ds_flow_zeta[:] = cor_flow
ds_windu_zeta[:] = cor_windu
ds_windv_zeta[:] = cor_windv

ds_salt_salt[:] = cor_gs_salt
ds_flow_salt[:] = cor_flow_salt
ds_salt_salt_hr1[:] = cor_gs_salt_hr1
ds_flow_salt_hr1[:] = cor_flow_salt_hr1

ds_temp_temp[:] = cor_gs_temp
ds_flow_temp[:] = cor_flow_temp
ds_airt_temp[:] = cor_airt
ds_temp_temp_hr1[:] = cor_gs_temp_hr1
ds_flow_temp_hr1[:] = cor_flow_temp_hr1
ds_airt_temp_hr1[:] = cor_airt_hr1

ds_out.close()

print("Write out maximum cross-correlations and associated lags")
lags = np.round(np.arange(-Nlag, Nlag+1) * dt_est,2)

print("Water Surface Elevations")
print("Steele Point (transport, fluvial inflow, wind u, wind v):")
i = 0
indx1 = np.where(np.abs(cor_tran[:,i]) == np.abs(cor_tran[:,i]).max())[0][0]
indx2 = np.where(np.abs(cor_flow[:,i]) == np.abs(cor_flow[:,i]).max())[0][0]
indx3 = np.where(np.abs(cor_windu[:,i]) == np.abs(cor_windu[:,i]).max())[0][0]
indx4 = np.where(np.abs(cor_windv[:,i]) == np.abs(cor_windv[:,i]).max())[0][0]
print(f"{np.round(cor_tran[indx1,i],3)}, {lags[indx1]} | {np.round(cor_flow[indx2,i],3)}, {lags[indx2]} | {np.round(cor_windu[indx3,i],3)}, {lags[indx3]} | {np.round(cor_windv[indx4,i],3)}, {lags[indx4]}")
print('--')
print(f"{np.round(cor_tran[indx1,i],3)}, {lags[indx1]}")

print("Speedy Point:")
i = 1
indx1 = np.where(np.abs(cor_tran[:,i]) == np.abs(cor_tran[:,i]).max())[0][0]
indx2 = np.where(np.abs(cor_flow[:,i]) == np.abs(cor_flow[:,i]).max())[0][0]
indx3 = np.where(np.abs(cor_windu[:,i]) == np.abs(cor_windu[:,i]).max())[0][0]
indx4 = np.where(np.abs(cor_windv[:,i]) == np.abs(cor_windv[:,i]).max())[0][0]
print(f"{np.round(cor_tran[indx1,i],3)}, {lags[indx1]} | {np.round(cor_flow[indx2,i],3)}, {lags[indx2]} | {np.round(cor_windu[indx3,i],3)}, {lags[indx3]} | {np.round(cor_windv[indx4,i],3)}, {lags[indx4]}")
print('--')
print(f"{np.round(cor_tran[indx1,i],3)}, {lags[indx1]} ")

print("========================================================================================")
print("Salinity")
print("Steele Point Top (GS Salt, Fluvial Inflow):")
i, j = 0,0
indx1 = np.where(np.abs(cor_gs_salt[:,i,j]) == np.abs(cor_gs_salt[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_salt[:,i,j]) == np.abs(cor_flow_salt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_salt[indx2,i,j],3)}, {lags[indx2]}")
print('--')
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]}")
print("Steele Point Bot:")
i, j = 0,1
indx1 = np.where(np.abs(cor_gs_salt[:,i,j]) == np.abs(cor_gs_salt[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_salt[:,i,j]) == np.abs(cor_flow_salt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_salt[indx2,i,j],3)}, {lags[indx2]}")
print('--')
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]}")
print("----------------")
print("Speedy Point Top:")
i, j = 1,0
indx1 = np.where(np.abs(cor_gs_salt[:,i,j]) == np.abs(cor_gs_salt[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_salt[:,i,j]) == np.abs(cor_flow_salt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_salt[indx2,i,j],3)}, {lags[indx2]}")
print('--')
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]}")
print("Speedy Point Bot:")
i, j = 1,1
indx1 = np.where(np.abs(cor_gs_salt[:,i,j]) == np.abs(cor_gs_salt[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_salt[:,i,j]) == np.abs(cor_flow_salt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_salt[indx2,i,j],3)}, {lags[indx2]}")
print('--')
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]}")
print("----------------")
print("HR1 Top:")
i, j = 2,0
indx1 = np.where(np.abs(cor_gs_salt[:,i,j]) == np.abs(cor_gs_salt[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_salt[:,i,j]) == np.abs(cor_flow_salt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_salt[indx2,i,j],3)}, {lags[indx2]}")

indx3 = np.where(np.abs(cor_gs_salt_hr1[:,j]) == np.abs(cor_gs_salt_hr1[:,j]).max())[0][0]
indx4 = np.where(np.abs(cor_flow_salt_hr1[:,j]) == np.abs(cor_flow_salt_hr1[:,j]).max())[0][0]
print(f"HR1 short: {np.round(cor_gs_salt_hr1[indx3,j],3)}, {lags[indx3]} | {np.round(cor_flow_salt_hr1[indx4,j],3)}, {lags[indx4]}")

print('--')
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]}")
print("HR1 Bot:")
i, j = 2,1
indx1 = np.where(np.abs(cor_gs_salt[:,i,j]) == np.abs(cor_gs_salt[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_salt[:,i,j]) == np.abs(cor_flow_salt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_salt[indx2,i,j],3)}, {lags[indx2]}")

indx3 = np.where(np.abs(cor_gs_salt_hr1[:,j]) == np.abs(cor_gs_salt_hr1[:,j]).max())[0][0]
indx4 = np.where(np.abs(cor_flow_salt_hr1[:,j]) == np.abs(cor_flow_salt_hr1[:,j]).max())[0][0]
print(f"HR1 short: {np.round(cor_gs_salt_hr1[indx3,j],3)}, {lags[indx3]} | {np.round(cor_flow_salt_hr1[indx4,j],3)}, {lags[indx4]}")

print('--')
print(f"{np.round(cor_gs_salt[indx1,i,j],3)}, {lags[indx1]}")

print("========================================================================================")
print("Temperature")
print("Steele Point Top (GS Temp, Fluvial Inflow, Air Temp):")
i, j = 0,0
indx1 = np.where(np.abs(cor_gs_temp[:,i,j]) == np.abs(cor_gs_temp[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_temp[:,i,j]) == np.abs(cor_flow_temp[:,i,j]).max())[0][0]
indx3 = np.where(np.abs(cor_airt[:,i,j]) == np.abs(cor_airt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_temp[indx2,i,j],3)}, {lags[indx2]} | {np.round(cor_airt[indx3,i,j],3)}, {lags[indx3]}")
print('--')
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]}")
print("Steele Point Bot:")
i, j = 0,1
indx1 = np.where(np.abs(cor_gs_temp[:,i,j]) == np.abs(cor_gs_temp[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_temp[:,i,j]) == np.abs(cor_flow_temp[:,i,j]).max())[0][0]
indx3 = np.where(np.abs(cor_airt[:,i,j]) == np.abs(cor_airt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_temp[indx2,i,j],3)}, {lags[indx2]} | {np.round(cor_airt[indx3,i,j],3)}, {lags[indx3]}")
print('--')
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]}")
print("----------------")
print("Speedy Point Top:")
i, j = 1,0
indx1 = np.where(np.abs(cor_gs_temp[:,i,j]) == np.abs(cor_gs_temp[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_temp[:,i,j]) == np.abs(cor_flow_temp[:,i,j]).max())[0][0]
indx3 = np.where(np.abs(cor_airt[:,i,j]) == np.abs(cor_airt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_temp[indx2,i,j],3)}, {lags[indx2]} | {np.round(cor_airt[indx3,i,j],3)}, {lags[indx3]}")
print('--')
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]}")
print("Speedy Point Bot:")
i, j = 1,1
indx1 = np.where(np.abs(cor_gs_temp[:,i,j]) == np.abs(cor_gs_temp[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_temp[:,i,j]) == np.abs(cor_flow_temp[:,i,j]).max())[0][0]
indx3 = np.where(np.abs(cor_airt[:,i,j]) == np.abs(cor_airt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_temp[indx2,i,j],3)}, {lags[indx2]} | {np.round(cor_airt[indx3,i,j],3)}, {lags[indx3]}")
print('--')
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]}")
print("----------------")
print("HR1 Top:")
i, j = 2,0
indx1 = np.where(np.abs(cor_gs_temp[:,i,j]) == np.abs(cor_gs_temp[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_temp[:,i,j]) == np.abs(cor_flow_temp[:,i,j]).max())[0][0]
indx3 = np.where(np.abs(cor_airt[:,i,j]) == np.abs(cor_airt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_temp[indx2,i,j],3)}, {lags[indx2]} | {np.round(cor_airt[indx3,i,j],3)}, {lags[indx3]}")

indx3 = np.where(np.abs(cor_gs_temp_hr1[:,j]) == np.abs(cor_gs_temp_hr1[:,j]).max())[0][0]
indx4 = np.where(np.abs(cor_flow_temp_hr1[:,j]) == np.abs(cor_flow_temp_hr1[:,j]).max())[0][0]
print(f"HR1 short: {np.round(cor_gs_temp_hr1[indx3,j],3)}, {lags[indx3]} | {np.round(cor_flow_temp_hr1[indx4,j],3)}, {lags[indx4]}")

print('--')
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]}")
print("HR1 Bot:")
i, j = 2,1
indx1 = np.where(np.abs(cor_gs_temp[:,i,j]) == np.abs(cor_gs_temp[:,i,j]).max())[0][0]
indx2 = np.where(np.abs(cor_flow_temp[:,i,j]) == np.abs(cor_flow_temp[:,i,j]).max())[0][0]
indx3 = np.where(np.abs(cor_airt[:,i,j]) == np.abs(cor_airt[:,i,j]).max())[0][0]
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]} | {np.round(cor_flow_temp[indx2,i,j],3)}, {lags[indx2]} | {np.round(cor_airt[indx3,i,j],3)}, {lags[indx3]}")

indx3 = np.where(np.abs(cor_gs_temp_hr1[:,j]) == np.abs(cor_gs_temp_hr1[:,j]).max())[0][0]
indx4 = np.where(np.abs(cor_flow_temp_hr1[:,j]) == np.abs(cor_flow_temp_hr1[:,j]).max())[0][0]
print(f"HR1 short: {np.round(cor_gs_temp_hr1[indx3,j],3)}, {lags[indx3]} | {np.round(cor_flow_temp_hr1[indx4,j],3)}, {lags[indx4]}")

print('--')
print(f"{np.round(cor_gs_temp[indx1,i,j],3)}, {lags[indx1]}")
print("=====================================================================")
print(f"r95 is {max(r95)}")