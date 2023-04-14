from ligotools import readligo as rl
from ligotools import utils as ul
import pytest
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, iirdesign, zpk2tf, freqz
import h5py
import json
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

# read in data and extract parameters - code from losc tutorial
eventname = 'GW150914'
fnjson = "data/BBH_events_v3.json"
events = json.load(open(fnjson,"r"))

event = events[eventname]
fn_H1 = event['fn_H1']
fn_L1 = event['fn_L1']
fn_template = event['fn_template']
fs = event['fs']
tevent = event['tevent']
fband = event['fband']

strain_H1, time_H1, chan_dict_H1 = rl.loaddata('data/'+fn_H1, 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata('data/'+fn_L1, 'L1')
time = time_H1
dt = time[1] - time[0]
data = np.random.normal(1,10000,100)

# number of sample for the fast fourier transform:
NFFT = 4*fs
Pxx_H1, freqs = mlab.psd(strain_H1, Fs = fs, NFFT = NFFT)
Pxx_L1, freqs = mlab.psd(strain_L1, Fs = fs, NFFT = NFFT)

# we will use interpolations of the ASDs computed above for whitening:
psd_H1 = interp1d(freqs, Pxx_H1)
psd_L1 = interp1d(freqs, Pxx_L1)


f_template = h5py.File('data/'+fn_template, "r")
template_p, template_c = f_template["template"][...]
t_m1 = f_template["/meta"].attrs['m1']
t_m2 = f_template["/meta"].attrs['m2']
t_a1 = f_template["/meta"].attrs['a1']
t_a2 = f_template["/meta"].attrs['a2']
t_approx = f_template["/meta"].attrs['approx']
f_template.close()
# the template extends to roughly 16s, zero-padded to the 32s data length. The merger will be roughly 16s in.
template_offset = 16.


# test for whiten
def test_whiten():
    ''' 
    Check the type and the length of the output array.
    '''
    white_ht = ul.whiten(strain_H1, psd_H1, dt)
    assert type(white_ht) == np.ndarray
    assert len(white_ht) == len(strain_H1)

# test for write_wavfile
def test_write_wavfile():
    '''
    Check the type of the output.
    '''
    assert type(data) == np.ndarray
    assert type(fs) == int
    assert type(ul.write_wavfile(eventname, fs, data)) is not None

# test for reqshift
def test_reqshift():
    '''
    Check that the output is an array, and the same length as the input array.
    '''
    assert len(ul.reqshift(data, fshift=100, sample_rate=4096)) == len(data)
    assert type(ul.reqshift(data, fshift=100, sample_rate=4096)) == np.ndarray

#loaded from the tutorial notebook needed for plotting
fnjson = "data/BBH_events_v3.json"
events = json.load(open(fnjson,"r"))
eventname = 'GW150914'
event = events[eventname]
fs = event['fs']
tevent = event['tevent']
fn_H1 = "data/H-H1_LOSC_4_V2-1126259446-32.hdf5"
strain_H1, time_H1, chan_dict_H1 = rl.loaddata(fn_H1, 'H1')
fn_L1 = "data/L-L1_LOSC_4_V2-1126259446-32.hdf5"
strain_L1, time_L1, chan_dict_L1 = rl.loaddata(fn_L1, 'L1')
time = time_H1
dt = time_H1[1] - time_H1[0]
dets = ['H1', 'L1']
#sample data
data = np.random.normal(0,10000,100)
NoneType = type(None)


# test for plotting function
def test_plotfunc():
    assert type(eventname) == str
    assert type(time) == np.ndarray
    assert type(dets[0]) == str
    assert type(tevent) == float