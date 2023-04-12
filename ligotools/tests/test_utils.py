import numpy as np
import json
from ligotools import *
import pytest

#loaded from the tutorial notebook
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

# test for whiten
def test_whiten():
    assert type(strain_H1) == np.ndarray
    assert type(dt) == np.float64
     
# test for write_wavfile        
def test_write_wavfile():
    assert type(fs) == int
    assert type(data) == np.ndarray
    assert type(write_wavfile(eventname,fs,data)) == NoneType

# test for requshift    
def test_reqshift():
    assert type(reqshift(data,fshift=100,sample_rate=4096)) == np.ndarray
    assert len(reqshift(data,fshift=100,sample_rate=4096)) == len(data)

    
# test for plotting function
def test_plot():
    assert type(eventname) == str
    assert type(time) == np.ndarray
    assert type(dets[0]) == str
    assert type(tevent) == float