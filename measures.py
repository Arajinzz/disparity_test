import numpy as np
import math

def ssd(b1, b2):
    return np.sum((b1 - b2)**2)

def ad(b1, b2):
    return np.sum(np.absolute(b1 - b2))


def cc(b1, b2):
    return np.sum(b1*b2)


def nc(b1, b2):
    t = cc(b1, b2)
    return  t / math.sqrt(t)
