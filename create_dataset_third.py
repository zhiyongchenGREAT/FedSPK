# %%
import torch
import numpy as np

# %%
torch.__version__

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra

from pyroomacoustics.directivities import (
    DirectivityPattern,
    DirectionVector,
    CardioidFamily,
)

# %%
def room_properties(absorption, size):
    if absorption=='high':
        absor = 0.7
    elif absorption=='medium':
        absor = 0.3
    elif absorption=='low':
        absor = 0.1
    else:
        raise ValueError("The absorption parameter can only take values ['low', 'medium', 'high']")
    
    if size=='large':
        size_coef = 5.0
    elif size=='medium':
        size_coef = 2.5
    elif size=='small':
        size_coef = 1.0
    else:
        raise ValueError("The absorption parameter can only take values ['low', 'medium', 'high']")
    return absor, size_coef

# %%
def clip(signal, high, low):
    """Clip a signal from above at high and from below at low."""
    s = signal.copy()

    s[np.where(s > high)] = high
    s[np.where(s < low)] = low

    return s

def normalize(signal, bits=None):
    """
    normalize to be in a given range. The default is to normalize the maximum
    amplitude to be one. An optional argument allows to normalize the signal
    to be within the range of a given signed integer representation of bits.
    """

    s = signal.copy()

    s /= np.abs(s).max()

    # if one wants to scale for bits allocated
    if bits is not None:
        s *= 2 ** (bits - 1) - 1
        s = clip(s, 2 ** (bits - 1) - 1, -(2 ** (bits - 1)))

    return s
    
def to_wav(filename, fs, signals, norm=False, bitdepth=np.float):
    """
    Save all the signals to wav files.

    Parameters
    ----------
    filename: str
        the name of the file
    mono: bool, optional
        if true, records only the center channel floor(M / 2) (default
        `False`)
    norm: bool, optional
        if true, normalize the signal to fit in the dynamic range (default
        `False`)
    bitdepth: int, optional
        the format of output samples [np.int8/16/32/64 or np.float
        (default)]
    """
    from scipy.io import wavfile

    signal = signals.T  # each column is a channel

    float_types = [float, np.float, np.float32, np.float64]

    if bitdepth in float_types:
        bits = None
    elif bitdepth is np.int8:
        bits = 8
    elif bitdepth is np.int16:
        bits = 16
    elif bitdepth is np.int32:
        bits = 32
    elif bitdepth is np.int64:
        bits = 64
    else:
        raise NameError("No such type.")

    if norm:
        signal = normalize(signal, bits=bits)

    signal = np.array(signal, dtype=bitdepth)

    wavfile.write(filename, fs, signal)

# %% [markdown]
# # G1_3

# %%
import glob

# %%
a = glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/*')

# %%
b = a[900:950]

# %%
b_list = []
for i in b:
    b_list.extend(glob.glob(i+'/*'))

# %%
b_list

# %%
def generate_wave(path, out_path):
    absor, size_coef = room_properties('high', 'small')
    room = pra.ShoeBox(
        p=size_coef*np.array([4, 3, 2.5]),
        materials=pra.Material(absor),
        fs=16000,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)

    # specify signal source
    fs, signal = wavfile.read(path)

    # add source to 2D room
    room.add_source([1.,1.,1.], signal=signal)

    # add microphone array
    room.add_microphone(loc=[3.5,2.5,1.])

    room.simulate()
    
    to_wav(out_path, fs, room.mic_array.signals[0,:], norm=True, bitdepth=np.int16)

# %%
glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/id10143/ioty72DM3MQ/*')

# %%
# make directory G1_3
import os
os.mkdir('/nvme/zhiyong/fllearn_data/G1_3')

for i in b_list:
    c = glob.glob(i+'/*')
    for j in c:
        label = j.split('/')[-3]
        out_path = '/nvme/zhiyong/fllearn_data/G1_3/'
        generate_wave(j, out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1])
        with open('/nvme/zhiyong/fllearn_data/train_list_G1_3.txt', 'a') as f:
            f.write(label+' '+out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1]+'\n')

# %% [markdown]
# # G2_3

# %%
import glob

# %%
a = glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/*')

# %%
b = a[950:1000]

# %%
b_list = []
for i in b:
    b_list.extend(glob.glob(i+'/*'))

# %%
b_list

# %%
def generate_wave(path, out_path):
    absor, size_coef = room_properties('low', 'large')
    room = pra.ShoeBox(
        p=np.array([4*size_coef, 3*size_coef, 2.5]),
        materials=pra.Material(absor),
        fs=16000,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)

    # specify signal source
    fs, signal = wavfile.read(path)

    # add source to 2D room
    room.add_source([1.,1.,1.], signal=signal)

    # add microphone array
    room.add_microphone(loc=[17.,12.,1.])

    room.simulate()
    to_wav(out_path, fs, room.mic_array.signals[0,:], norm=True, bitdepth=np.int16)

# %%
glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/id10143/ioty72DM3MQ/*')

# %%
# make directory G2_3
import os
os.mkdir('/nvme/zhiyong/fllearn_data/G2_3')

for i in b_list:
    c = glob.glob(i+'/*')
    for j in c:
        label = j.split('/')[-3]
        out_path = '/nvme/zhiyong/fllearn_data/G2_3/'
        generate_wave(j, out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1])
        with open('/nvme/zhiyong/fllearn_data/train_list_G2_3.txt', 'a') as f:
            f.write(label+' '+out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1]+'\n')

# %% [markdown]
# # G3_3

# %%
import glob

# %%
a = glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/*')

# %%
b = a[1000:1050]

# %%
b_list = []
for i in b:
    b_list.extend(glob.glob(i+'/*'))

# %%
b_list

# %%
def generate_wave(path, out_path):
    absor, size_coef = room_properties('medium', 'medium')
    room = pra.ShoeBox(
        p=np.array([4*size_coef, 3*size_coef, 2.5]),
        materials=pra.Material(absor),
        fs=16000,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)

    # specify signal source
    fs, signal = wavfile.read(path)

    # add source to 2D room
    room.add_source([1.,1.,1.], signal=signal)

    # add microphone array
    room.add_microphone(loc=[8.,6.,1.])

    room.simulate()
    to_wav(out_path, fs, room.mic_array.signals[0,:], norm=True, bitdepth=np.int16)

# %%
glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/id10143/ioty72DM3MQ/*')

# %%
# make directory G3_3
import os
os.mkdir('/nvme/zhiyong/fllearn_data/G3_3')

for i in b_list:
    c = glob.glob(i+'/*')
    for j in c:
        label = j.split('/')[-3]
        out_path = '/nvme/zhiyong/fllearn_data/G3_3/'
        generate_wave(j, out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1])
        with open('/nvme/zhiyong/fllearn_data/train_list_G3_3.txt', 'a') as f:
            f.write(label+' '+out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1]+'\n')

# %% [markdown]
# # G4_3

# %%
import glob

# %%
a = glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/*')

# %%
b = a[1050:1100]

# %%
b_list = []
for i in b:
    b_list.extend(glob.glob(i+'/*'))

# %%
b_list

# %%
def generate_wave(path, out_path):
    absor, size_coef = room_properties('medium', 'medium')
    room = pra.ShoeBox(
        p=np.array([4*size_coef, 3*size_coef, 2.5]),
        materials=pra.Material(absor),
        fs=16000,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)

    # specify signal source
    fs, signal = wavfile.read(path)

    # add source to 2D room
    room.add_source([1.,1.,1.], signal=signal)

    # add noise source
    fs, noise = wavfile.read("/nvme/zhiyong/musan/noise/free-sound/noise-free-sound-0000.wav")

    room.add_source([4.,7.,1.], signal=noise[:len(signal)], delay=0.)

    # add microphone array
    room.add_microphone(loc=[8.,4.,1.])

    room.simulate()
    to_wav(out_path, fs, room.mic_array.signals[0,:], norm=True, bitdepth=np.int16)

# %%
glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/id10143/ioty72DM3MQ/*')

# %%
# make directory G4_3
import os
os.mkdir('/nvme/zhiyong/fllearn_data/G4_3')

for i in b_list:
    c = glob.glob(i+'/*')
    for j in c:
        label = j.split('/')[-3]
        out_path = '/nvme/zhiyong/fllearn_data/G4_3/'
        generate_wave(j, out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1])
        with open('/nvme/zhiyong/fllearn_data/train_list_G4_3.txt', 'a') as f:
            f.write(label+' '+out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1]+'\n')

# %% [markdown]
# # G5_3

# %%
import glob

# %%
a = glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/*')

# %%
b = a[1100:1150]

# %%
b_list = []
for i in b:
    b_list.extend(glob.glob(i+'/*'))

# %%
b_list

# %%
def generate_wave(path, out_path):
    fft_len = 512
    Lg_t = 0.100                # filter size in seconds
    Lg = np.ceil(Lg_t*16000)       # in samples
    absor, size_coef = room_properties('medium', 'medium')
    room = pra.ShoeBox(
        p=np.array([4*size_coef, 3*size_coef, 2.5]),
        materials=pra.Material(absor),
        fs=16000,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)

    # specify signal source
    fs, signal = wavfile.read(path)

    # add source to 2D room
    room.add_source([1.,1.,1.], signal=signal)

    # add microphone array
    R = pra.circular_2D_array(center=[8.,2.], M=6, phi0=0, radius=0.05)
    R = np.concatenate((R, np.ones((1, 6))))
    # dir_list = []
    # for i in range(6):
    #     dir_list.append(CardioidFamily(
    #     orientation=DirectionVector(azimuth=0, colatitude=90, degrees=True),
    #     pattern_enum=DirectivityPattern.OMNI,))

    # mics = pra.MicrophoneArray(R, room.fs, directivity=dir_list)
    mics = pra.Beamformer(R, fs, N=fft_len, Lg=Lg)
    room.add_microphone_array(mics)

    # Compute DAS weights
    mics.rake_delay_and_sum_weights(room.sources[0][:1])

    room.simulate()
    signal_das = mics.process(FD=False)
    to_wav(out_path, fs, signal_das, norm=True, bitdepth=np.int16)

# %%
glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/id10143/ioty72DM3MQ/*')

# %%
# make directory G5_3
import os
os.mkdir('/nvme/zhiyong/fllearn_data/G5_3')

for i in b_list:
    c = glob.glob(i+'/*')
    for j in c:
        label = j.split('/')[-3]
        out_path = '/nvme/zhiyong/fllearn_data/G5_3/'
        generate_wave(j, out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1])
        with open('/nvme/zhiyong/fllearn_data/train_list_G5_3.txt', 'a') as f:
            f.write(label+' '+out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1]+'\n')

# %%


# %% [markdown]
# # G6_3

# %%
import glob

# %%
a = glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/*')

# %%
b = a[1150:1200]

# %%
b_list = []
for i in b:
    b_list.extend(glob.glob(i+'/*'))

# %%
b_list

# %%
def generate_wave(path, out_path):
    fft_len = 512
    Lg_t = 0.100                # filter size in seconds
    Lg = np.ceil(Lg_t*16000)       # in samples
    absor, size_coef = room_properties('medium', 'medium')
    room = pra.ShoeBox(
        p=np.array([4*size_coef, 3*size_coef, 2.5]),
        materials=pra.Material(absor),
        fs=16000,
        max_order=3,
        ray_tracing=True,
        air_absorption=True,
    )
    room.set_ray_tracing(receiver_radius=0.1, n_rays=10000, energy_thres=1e-5)

    # specify signal source
    fs, signal = wavfile.read(path)

    # add source to 2D room
    room.add_source([1.,1.,1.], signal=signal)

    # add noise source
    fs, noise = wavfile.read("/nvme/zhiyong/musan/noise/free-sound/noise-free-sound-0000.wav")

    room.add_source([8.,7.,1.], signal=noise[:len(signal)], delay=0.)

    # add microphone array
    R = pra.circular_2D_array(center=[8.,2.], M=6, phi0=0, radius=0.05)
    R = np.concatenate((R, np.ones((1, 6))))

    mics = pra.Beamformer(R, fs, N=fft_len, Lg=Lg)
    room.add_microphone_array(mics)

    # Compute DAS weights
    mics.rake_delay_and_sum_weights(room.sources[0][:1])

    room.simulate()
    signal_das = mics.process(FD=False)
    to_wav(out_path, fs, signal_das, norm=True, bitdepth=np.int16)

# %%
glob.glob('/nvme/zhiyong/voxceleb/vox1/dev/wav/id10143/ioty72DM3MQ/*')

# %%
# make directory G6_3
import os
os.mkdir('/nvme/zhiyong/fllearn_data/G6_3')

for i in b_list:
    c = glob.glob(i+'/*')
    for j in c:
        label = j.split('/')[-3]
        out_path = '/nvme/zhiyong/fllearn_data/G6_3/'
        generate_wave(j, out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1])
        with open('/nvme/zhiyong/fllearn_data/train_list_G6_3.txt', 'a') as f:
            f.write(label+' '+out_path+label+'_'+j.split('/')[-2]+'_'+j.split('/')[-1]+'\n')


