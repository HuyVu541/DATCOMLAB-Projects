import pandas as pd
import numpy as np
from numpy.linalg import norm
 
def cosine(window1, window2, window):
    window1 = window1.to_numpy()
    window2 = window2.to_numpy()
    cosine = np.dot(window1, window2)/(norm(window1)*norm(window2))
    return cosine