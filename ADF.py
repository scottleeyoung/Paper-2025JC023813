from statsmodels.tsa.stattools import adfuller
import xarray as xr

print("Reading detrended time series")
ds = xr.open_dataset("data/detrended_driver_timeseries.nc")
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

print("Conducting Augmented Dickey-Fuller Test using the python module")
print("Water Levels")
print(adfuller(zeta[:,0], regression='n', autolag='AIC'), adfuller(zeta[:,1], regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("Salt Top")
print(adfuller(salt[:,0,0], regression='n', autolag='AIC'), adfuller(salt[:,1,0], regression='n', autolag='AIC'), 
      adfuller(salt[:,2,0], regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("Salt Bot")
print(adfuller(salt[:,0,1], regression='n', autolag='AIC'), adfuller(salt[:,1,1], regression='n', autolag='AIC'), 
      adfuller(salt[:,2,1], regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("Temp Top")
print(adfuller(temp[:,0,0], regression='n', autolag='AIC'), adfuller(temp[:,1,0], regression='n', autolag='AIC'), 
      adfuller(temp[:,2,0], regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("Temp Bot")
print(adfuller(temp[:,0,1], regression='n', autolag='AIC'), adfuller(temp[:,1,1], regression='n', autolag='AIC'), 
      adfuller(temp[:,2,1], regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("GS Tran")
print(adfuller(gs_tran, regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("GS Temp")
print(adfuller(gs_temp, regression='n', autolag='AIC'))

print("=====================")
print("GS Salt")
print(adfuller(gs_salt, regression='n', autolag='AIC'))

print("=====================")
print("Fluvial Inflow")
print(adfuller(flow, regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("Air Temperature")
print(adfuller(airt, regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("Eastward Winds")
print(adfuller(wind_u, regression='n', autolag='AIC'))
print(" ")
print("=====================")
print("Northward Winds")
print(adfuller(wind_v, regression='n', autolag='AIC'))