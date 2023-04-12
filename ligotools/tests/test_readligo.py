from ligotools import readligo as rl
from ligotools import utils
import pytest
import numpy as np

# loading the data
strain_H1, time_H1, chan_dict_H1 = rl.loaddata("data/H-H1_LOSC_4_V2-1126259446-32.hdf5", 'H1')
strain_L1, time_L1, chan_dict_L1 = rl.loaddata("data/L-L1_LOSC_4_V2-1126259446-32.hdf5", 'L1')

# Read_Hdf5
strain, gpsStart, ts, qmask, shortnameList, injmask, injnameList = rl.read_hdf5("data/H-H1_LOSC_4_V2-1126259446-32.hdf5")

# test 1 for readligo.py
def test_h1_empty():
    assert strain_H1.any()
    assert time_H1.any()
    assert chan_dict_H1 is not None
    
# test 2 for readligo.py
def test_l1_empty():
    assert strain_L1.any()
    assert time_L1.any()
    assert chan_dict_L1 is not None
    
# test 3 for readligo.py
def test_h1_vals():
    assert strain_H1[0] == 2.177040281449375e-19
    assert time_H1[1] == 1126259446.0002441
    assert chan_dict_H1['DATA'][2] == 1

# test 4 for readligo.py
def test_l1_vals():
    assert strain_L1[0] == -1.0428999418774637e-18
    assert time_L1[1] == 1126259446.0002441
    assert chan_dict_L1['DATA'][2] == 1

