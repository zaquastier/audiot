import numpy as np

def frequency_to_index(support, f):
    """
    Return support index associated with f (closest to bottom).

    Args:
        support (np.ndarray): Frequency support (in Hz).
        f (double): Frequency (in Hz).

    Returns:
        index (int): closest to bottom frequency index
    """

    if f < 0:
        return -1
    
    index = np.argmax(support > f)

    return index