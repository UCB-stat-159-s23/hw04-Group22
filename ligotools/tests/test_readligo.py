import numpy as np
import json
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from ligotools import readligo as rl
import pytest


strain_H1, time_H1, chan_dict_H1 = rl.loaddata("data/H-H1_LOSC_4_V2-1126259446-32.hdf5", 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata("data/L-L1_LOSC_4_V2-1126259446-32.hdf5", 'L1')

strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = rl.read_hdf5("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")

#test 1 for readligo.py
def test_loaddata():
    assert type(strain_L1)==np.ndarray and type(time_L1)==np.ndarray and type(chan_dict_L1)==dict
    assert type(strain_H1)==np.ndarray and type(time_H1)==np.ndarray and type(chan_dict_H1)==dict
    
#test 2 for readligo.py
def test_l1_empty():
    assert strain_L1.any()
    assert time_L1.any()
    assert chan_dict_L1 is not None
    
#test 3 for readligo.py
def test_h1_empty():
    assert strain_H1.any()
    assert time_H1.any()
    assert chan_dict_H1 is not None

#test 4 for readligo.py
def test_dimensions():
    assert len(strain_H1) == 131072
    assert len(strain_L1) == 131072
    assert len(time_H1) == 131072
    assert len(time_L1) == 131072
    assert len(chan_dict_H1) == 13
    assert len(chan_dict_L1) == 13
    
