import numpy as np
import scipy.stats as st
 
class Preprocessing:
       
    def build_design_matrix(X):
        x0 = np.array([np.ones(len(X))]).T
        X = np.column_stack([x0,X[:,0]])
        return X