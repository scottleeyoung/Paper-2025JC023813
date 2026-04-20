import numpy as np
import netCDF4 as nc
import xarray as xr

import sys
sys.path.insert(0,"../modules")
import romspy

def wavelet(array, dt=1, dj=0.25, c=6, J=None, s=[1,2]):
    '''
    This function calculates the wavelet transform of a time series using the Morlet wavelet in accordance with the method outlined in
    Torrence and Compo 1998.
    '''
    import numpy as np

    N = len(array)

    # normalizing time series
    array = array / np.var(array)

    # taking the FFT of the time series
    X = np.fft.fft(array)

    # creating the radian frequency array
    k = np.arange(N)
    w_pos = (2 * np.pi * k[:N//2 + 1]) / (N * dt)
    w_neg = np.sort((-2 * np.pi * k[N//2 + 1:]) / (N * dt))
    w = np.concatenate([w_pos, w_neg])

    # creating the scale array
    if s == [1,2]:
        s0 = 2 * dt
        if J == None:
            J = (1 / dj) * np.log2(N * dt / s0)
        j = np.arange(0,J+1)
        s = s0 * 2 ** (j *dj)
    
    # calculating the equivalent fourier period
    alpha = (4 * np.pi) / (c + np.sqrt(2 + c ** 2))
    F_per = alpha * s

    # creating the Heaviside function
    H = np.array(w > 0, dtype=float)
    
    # carrying out the Wavelet transform in the Fourier space
    M = len(s)
    n = np.arange(N)
    W = np.array([[99] * N] * M, dtype=complex)
    for i in range(M):
        norm = np.sqrt(w[1] * s[i] * N) / np.pi ** 0.25
        psi = norm * H * np.conj(np.exp(-0.5 * (s[i] * w - c) ** 2))     # normalized Fourier transformed wavelet
        W_hat = X * psi 
        W[i, :] = np.fft.ifft(W_hat)
    
    # removing zero padding
    W = W[:, :N]

    # calculating the cone of influence
    coi = (N / 2 - np.abs(np.arange(0, N) - (N - 1) / 2))
    coi = alpha * dt * coi / np.sqrt(2)

    return W, s, F_per, coi

def xwt_sig_level(x, y, dt=1, dj=0.25, c=6, J=None, s=[1,2], sig_lvl=95, N=1000):
    xW_power = []
    for i in range(N):
        x_noise = np.random.normal(loc=x.mean(), scale=np.std(x), size=x.size)
        y_noise = np.random.normal(loc=y.mean(), scale=np.std(y), size=y.size)
        W1, s1, _, _ = wavelet(x_noise, dt, dj, c, J, s)
        W2, _, _, _ = wavelet(y_noise, dt, dj, c, J, s)
        W3 = W1 * np.conj(W2)
        W3 = np.abs(W3) / s1[:,np.newaxis]
        xW_power.append(W3)
        romspy.reprint(f"{np.round(100 * (i+1) / N,2)} %")
    return np.percentile(np.array(xW_power), sig_lvl)

def xwavelet(x, y, dt=1, dj=0.25, c=6, J=None, s=[1,2], sig_lvl=None, N=1000):
    x = (x - x.mean()) / np.std(x)
    y = (y - y.mean()) / np.std(y)
    W1, s1, Fper, coi = wavelet(x, dt=dt, dj=dj, c=c, J=J, s=s)
    W2, _, _, _ = wavelet(y, dt=dt, dj=dj, c=c, J=J, s=s)
    W3 = W1 * np.conj(W2)
    W3 = W3 / s1[:,np.newaxis]

    if sig_lvl != None:
        sig = xwt_sig_level(x, y, dt=dt, dj=dj, c=c, J=J, s=s, sig_lvl=sig_lvl, N=N)
        return W3, s1, Fper, coi, sig
    else:
        return W3, s1, Fper, coi, 10
    
print("Reading detrended time series")
dt = 1 #15/1440

ds = xr.open_dataset("data/detrended_driver_timeseries.nc")
# flow = ds['flow'].values
# airt = ds['airt'].values
gs_salt = ds['gs_salt'].values[::96]
gs_temp = ds['gs_temp'].values[::96]
gs_tran = ds['gs_tran'].values[::96]
# wind_u = ds['wind_u'].values
# wind_v = ds['wind_v'].values
ds.close()

ds = xr.open_dataset("data/detrended_estuary_timeseries.nc")
time = ds['time'].values[::96]
zeta = ds['zeta'].values[::96,:]
salt = ds['salt'].values[::96,:,:]
temp = ds['temp'].values[::96,:,:]
ds.close()

print("Water Levels cross-Wavelets")
sig_lvl = 95

xW_tran0, s_tran0, Fper_tran0, coi_tran0, sig_tran0 = xwavelet(gs_tran, zeta[:,0], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_tran0)
phase[np.where(np.abs(xW_tran0) < sig_tran0)] = np.nan
u0 = np.cos(phase) 
v0 = np.sin(phase)

xW_tran1, s_tran1, Fper_tran1, coi_tran1, sig_tran1 = xwavelet(gs_tran, zeta[:,1], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_tran1)
phase[np.where(np.abs(xW_tran1) < sig_tran1)] = np.nan
u1 = np.cos(phase) 
v1 = np.sin(phase)

print("Salinity cross-Wavelets")
sig_lvl = 95
print("00")
xW_gs_salt00, s_gs_salt00, Fper_gs_salt00, coi_gs_salt00, sig_gs_salt00 = xwavelet(gs_salt, salt[:,0,0], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_salt00)
phase[np.where(np.abs(xW_gs_salt00) < sig_gs_salt00)] = np.nan
u_gs_salt00 = np.cos(phase) 
v_gs_salt00 = np.sin(phase)

print("01")
xW_gs_salt01, s_gs_salt01, Fper_gs_salt01, coi_gs_salt01, sig_gs_salt01 = xwavelet(gs_salt, salt[:,0,1], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_salt01)
phase[np.where(np.abs(xW_gs_salt01) < sig_gs_salt01)] = np.nan
u_gs_salt01 = np.cos(phase) 
v_gs_salt01 = np.sin(phase)

print("10")
xW_gs_salt10, s_gs_salt10, Fper_gs_salt10, coi_gs_salt10, sig_gs_salt10 = xwavelet(gs_salt, salt[:,1,0], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_salt10)
phase[np.where(np.abs(xW_gs_salt10) < sig_gs_salt10)] = np.nan
u_gs_salt10 = np.cos(phase) 
v_gs_salt10 = np.sin(phase)

print("11")
xW_gs_salt11, s_gs_salt11, Fper_gs_salt11, coi_gs_salt11, sig_gs_salt11 = xwavelet(gs_salt, salt[:,1,1], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_salt11)
phase[np.where(np.abs(xW_gs_salt11) < sig_gs_salt11)] = np.nan
u_gs_salt11 = np.cos(phase) 
v_gs_salt11 = np.sin(phase)

print("20")
xW_gs_salt20, s_gs_salt20, Fper_gs_salt20, coi_gs_salt20, sig_gs_salt20 = xwavelet(gs_salt, salt[:,2,0], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_salt20)
phase[np.where(np.abs(xW_gs_salt20) < sig_gs_salt20)] = np.nan
u_gs_salt20 = np.cos(phase) 
v_gs_salt20 = np.sin(phase)

print("21")
xW_gs_salt21, s_gs_salt21, Fper_gs_salt21, coi_gs_salt21, sig_gs_salt21 = xwavelet(gs_salt, salt[:,2,1], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_salt21)
phase[np.where(np.abs(xW_gs_salt21) < sig_gs_salt21)] = np.nan
u_gs_salt21 = np.cos(phase) 
v_gs_salt21 = np.sin(phase)

print("Temperature cross-Wavelets")
sig_lvl = 95
print("00")
xW_gs_temp00, s_gs_temp00, Fper_gs_temp00, coi_gs_temp00, sig_gs_temp00 = xwavelet(gs_temp, temp[:,0,0], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_temp00)
phase[np.where(np.abs(xW_gs_temp00) < sig_gs_temp00)] = np.nan
u_gs_temp00 = np.cos(phase) 
v_gs_temp00 = np.sin(phase)

print("01")
xW_gs_temp01, s_gs_temp01, Fper_gs_temp01, coi_gs_temp01, sig_gs_temp01 = xwavelet(gs_temp, temp[:,0,1], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_temp01)
phase[np.where(np.abs(xW_gs_temp01) < sig_gs_temp01)] = np.nan
u_gs_temp01 = np.cos(phase) 
v_gs_temp01 = np.sin(phase)

print("10")
xW_gs_temp10, s_gs_temp10, Fper_gs_temp10, coi_gs_temp10, sig_gs_temp10 = xwavelet(gs_temp, temp[:,1,0], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_temp10)
phase[np.where(np.abs(xW_gs_temp10) < sig_gs_temp10)] = np.nan
u_gs_temp10 = np.cos(phase) 
v_gs_temp10 = np.sin(phase)

print("11")
xW_gs_temp11, s_gs_temp11, Fper_gs_temp11, coi_gs_temp11, sig_gs_temp11 = xwavelet(gs_temp, temp[:,1,1], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_temp11)
phase[np.where(np.abs(xW_gs_temp11) < sig_gs_temp11)] = np.nan
u_gs_temp11 = np.cos(phase) 
v_gs_temp11 = np.sin(phase)

print("20")
xW_gs_temp20, s_gs_temp20, Fper_gs_temp20, coi_gs_temp20, sig_gs_temp20 = xwavelet(gs_temp, temp[:,2,0], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_temp20)
phase[np.where(np.abs(xW_gs_temp20) < sig_gs_temp20)] = np.nan
u_gs_temp20 = np.cos(phase) 
v_gs_temp20 = np.sin(phase)

print("21")
xW_gs_temp21, s_gs_temp21, Fper_gs_temp21, coi_gs_temp21, sig_gs_temp21 = xwavelet(gs_temp, temp[:,2,1], dt=dt, dj=1/12, sig_lvl=sig_lvl)
phase = np.angle(xW_gs_temp21)
phase[np.where(np.abs(xW_gs_temp21) < sig_gs_temp21)] = np.nan
u_gs_temp21 = np.cos(phase) 
v_gs_temp21 = np.sin(phase)

print("Writing data to file")
ds_out = nc.Dataset("data/xwavelet.nc", mode='w', format='NETCDF4')
ds_out.createDimension("time", xW_tran0.shape[1])
ds_out.createDimension("s", xW_tran0.shape[0])
ds_out.createDimension("one", 1)
ds_out.createDimension("two", 2)
ds_out.createDimension("three", 3)

ds_time = ds_out.createVariable("time", "f8", ("time",))
ds_scales = ds_out.createVariable("scales", "f8", ("s",))
ds_period = ds_out.createVariable("periods", "f8", ("s",))
ds_coi_tran = ds_out.createVariable("coi_tran", "f8", ("time", "two",))
ds_coi_salt = ds_out.createVariable("coi_salt", "f8", ("time", "three", "two"))
ds_coi_temp = ds_out.createVariable("coi_temp", "f8", ("time", "three", "two"))
ds_tran = ds_out.createVariable("xW_tran", "f8", ("s","time","two",))
ds_salt = ds_out.createVariable("xW_salt", "f8", ("s","time", "three", "two"))
ds_temp = ds_out.createVariable("xW_temp", "f8", ("s","time", "three", "two"))
ds_tran_sig = ds_out.createVariable("xW_tran_sig", "f8", ("two",))
ds_salt_sig = ds_out.createVariable("xW_salt_sig", "f8", ("three", "two"))
ds_temp_sig = ds_out.createVariable("xW_temp_sig", "f8", ("three", "two"))
ds_tran_u = ds_out.createVariable("u_phase_tran", "f8", ("s","time","two",))
ds_salt_u = ds_out.createVariable("u_phase_salt", "f8", ("s","time", "three", "two"))
ds_temp_u = ds_out.createVariable("u_phase_temp", "f8", ("s","time", "three", "two"))
ds_tran_v = ds_out.createVariable("v_phase_tran", "f8", ("s","time","two",))
ds_salt_v = ds_out.createVariable("v_phase_salt", "f8", ("s","time", "three", "two"))
ds_temp_v = ds_out.createVariable("v_phase_temp", "f8", ("s","time", "three", "two"))

ds_time[:] = time
ds_scales[:] = s_tran0
ds_period[:] = Fper_tran0
ds_coi_tran[:,0] = coi_tran0
ds_coi_tran[:,1] = coi_tran1
ds_tran[:,:,0] = np.abs(xW_tran0)
ds_tran[:,:,1] = np.abs(xW_tran1)
ds_tran_sig[0] = sig_tran0
ds_tran_sig[1] = sig_tran1
ds_tran_u [:,:,0] = u0
ds_tran_u [:,:,1] = u1
ds_tran_v [:,:,0] = v0
ds_tran_v [:,:,1] = v1

ds_coi_salt[:,0,0] = coi_gs_salt00
ds_coi_salt[:,0,1] = coi_gs_salt01
ds_coi_salt[:,1,0] = coi_gs_salt00
ds_coi_salt[:,1,1] = coi_gs_salt01
ds_coi_salt[:,2,0] = coi_gs_salt20
ds_coi_salt[:,2,1] = coi_gs_salt21
ds_salt[:,:,0,0] = np.abs(xW_gs_salt00)
ds_salt[:,:,0,1] = np.abs(xW_gs_salt01)
ds_salt[:,:,1,0] = np.abs(xW_gs_salt10)
ds_salt[:,:,1,1] = np.abs(xW_gs_salt11)
ds_salt[:,:,2,0] = np.abs(xW_gs_salt20)
ds_salt[:,:,2,1] = np.abs(xW_gs_salt21)
ds_salt_sig[0,0] = sig_gs_salt00
ds_salt_sig[0,1] = sig_gs_salt01
ds_salt_sig[1,0] = sig_gs_salt10
ds_salt_sig[1,1] = sig_gs_salt11
ds_salt_sig[2,0] = sig_gs_salt20
ds_salt_sig[2,1] = sig_gs_salt21
ds_salt_u [:,:,0,0] = u_gs_salt00
ds_salt_u [:,:,0,1] = u_gs_salt01
ds_salt_u [:,:,1,0] = u_gs_salt10
ds_salt_u [:,:,1,1] = u_gs_salt11
ds_salt_u [:,:,2,0] = u_gs_salt20
ds_salt_u [:,:,2,1] = u_gs_salt21
ds_salt_v [:,:,0,0] = v_gs_salt00
ds_salt_v [:,:,0,1] = v_gs_salt01
ds_salt_v [:,:,1,0] = v_gs_salt10
ds_salt_v [:,:,1,1] = v_gs_salt11
ds_salt_v [:,:,2,0] = v_gs_salt20
ds_salt_v [:,:,2,1] = v_gs_salt21

ds_coi_temp[:,0,0] = coi_gs_temp00
ds_coi_temp[:,0,1] = coi_gs_temp01
ds_coi_temp[:,1,0] = coi_gs_temp00
ds_coi_temp[:,1,1] = coi_gs_temp01
ds_coi_temp[:,2,0] = coi_gs_temp20
ds_coi_temp[:,2,1] = coi_gs_temp21
ds_temp[:,:,0,0] = np.abs(xW_gs_temp00)
ds_temp[:,:,0,1] = np.abs(xW_gs_temp01)
ds_temp[:,:,1,0] = np.abs(xW_gs_temp10)
ds_temp[:,:,1,1] = np.abs(xW_gs_temp11)
ds_temp[:,:,2,0] = np.abs(xW_gs_temp20)
ds_temp[:,:,2,1] = np.abs(xW_gs_temp21)
ds_temp_sig[0,0] = sig_gs_temp00
ds_temp_sig[0,1] = sig_gs_temp01
ds_temp_sig[1,0] = sig_gs_temp10
ds_temp_sig[1,1] = sig_gs_temp11
ds_temp_sig[2,0] = sig_gs_temp20
ds_temp_sig[2,1] = sig_gs_temp21
ds_temp_u [:,:,0,0] = u_gs_temp00
ds_temp_u [:,:,0,1] = u_gs_temp01
ds_temp_u [:,:,1,0] = u_gs_temp10
ds_temp_u [:,:,1,1] = u_gs_temp11
ds_temp_u [:,:,2,0] = u_gs_temp20
ds_temp_u [:,:,2,1] = u_gs_temp21
ds_temp_v [:,:,0,0] = v_gs_temp00
ds_temp_v [:,:,0,1] = v_gs_temp01
ds_temp_v [:,:,1,0] = v_gs_temp10
ds_temp_v [:,:,1,1] = v_gs_temp11
ds_temp_v [:,:,2,0] = v_gs_temp20
ds_temp_v [:,:,2,1] = v_gs_temp21

ds_out.close()