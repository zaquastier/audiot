import numpy as np

def euclidean(f1, f2):
    """
    Euclidean distance between two frequencies.

    Args:
        f1 (double): frequency
        f2 (double): frequency

    Returns:
        euclidean distance between f1 and f2
    """
    return np.abs(f1 - f2)

def l2(f1, f2):
    """
    Euclidean distance between two frequencies.

    Args:
        f1 (double): frequency
        f2 (double): frequency

    Returns:
        euclidean distance between f1 and f2
    """
    return (f1 - f2)**2