import numpy as np
import librosa

from .utils import *

def load_signal(path, sr=44100, duration=None, smooth_len=0):
    """ 
    Load audio signal.

    Args:
        path (string): file to path
        sr (int): sample rate
        duration (double): in seconds
        smooth_len (int): size in samples of smoothing
    
    Returns:
        audio_signal (np.ndarray): audio signal as an array
    """


    signal, _ = librosa.load(path, sr=sr, duration=duration)

    if smooth_len > 0:
        smoothing = np.linspace(0, 1, smooth_len)
        signal[:smooth_len] *= smoothing
        signal[-smooth_len:] *= np.flip(smoothing)

    return signal

def fft(signal, sr=44100, return_support=False, max_frequency=None):
    """
    Compute the complex FFT.

    Args:
        signal (np.ndarray): audio signal
        sr (int): sample rate
        return_support (bool): if true returns frequency support
        max_frequency (int): truncate fft up to a max frequency (in Hz)

    Returns:
        spectrum_normalized (np.ndarray): normalized fft of signal
        (if return_support) support (np.ndarray): frequency support

    """

    spectrum = np.fft.rfft(signal)

    support = np.fft.rfftfreq(len(signal), d=1/sr)

    if max_frequency:
        index = frequency_to_index(support, max_frequency)
        support = support[:index]
        spectrum = spectrum[:index]

    if return_support:
        return spectrum, support
    
    return spectrum

def fft_normalized(signal, sr=44100, return_support=False, max_frequency=None):
    """
    Compute normalized FFT.

    Args:
        signal (np.ndarray): audio signal
        sr (int): sample rate
        return_support (bool): if true returns frequency support
        max_frequency (int): truncate fft up to a max frequency (in Hz)

    Returns:
        spectrum_normalized (np.ndarray): normalized fft of signal
        (if return_support) support (np.ndarray): frequency support

    """

    spectrum = np.fft.rfft(signal)
    spectrum = np.abs(spectrum)

    support = np.fft.rfftfreq(len(signal), d=1/sr)

    if max_frequency:
        index = frequency_to_index(support, max_frequency)
        support = support[:index]
        spectrum = spectrum[:index]

    spectrum_normalized = spectrum / np.sum(spectrum)

    if return_support:
        return spectrum_normalized, support
    

    return spectrum_normalized