import pandas as pd
import numpy as np

def get_window(time_series, window_len):
    
    X = []
    for i in range(len(time_series)-window_len):
        X.append(time_series[i:i+window_len])
        
    return np.array(X)