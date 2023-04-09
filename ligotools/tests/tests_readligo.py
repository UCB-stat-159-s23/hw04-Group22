from ligotools import readligo as rl
from ligotools import utils
import pytest
import numpy as np

# Loaddata
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

# test whiten function in utils.py
def test_whiten():
    # Generate a random strain signal with known properties
    Nt = 4096
    dt = 0.001
    strain = np.random.randn(Nt)

    # Generate an arbitrary PSD to use for whitening
    interp_psd = lambda f: 0.5 / ((f / 20.0)**4 + 1.0)  # arbitrary PSD

    # Whiten the signal
    white_ht = whiten(strain, interp_psd, dt)

    # Check that the length of the whitened signal matches the input
    assert len(white_ht) == len(strain)

# test write_wavfile function in utils.py



# test reqshift function in utils.py
def test_reqshift():
    data = np.random.randn(1000)
    shifted_data = reqshift(data, fshift=50, sample_rate=2000)
    assert len(shifted_data) == len(data)
    assert np.allclose(np.abs(np.fft.rfft(data)), np.abs(np.fft.rfft(shifted_data)))