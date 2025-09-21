import numpy as np
import scipy.stats as st
from typing import Tuple

class DataGenerator():
    def __init__(self, n,w, x_min, x_max, std=1):
        self.n = n # 1000
        self.w = w # weights matrix
        self.x_min = x_min
        self.x_max = x_max
        self.std = std
        pass

    def get_data(self):
        x0 = np.ones((self.n,1))    # coluna de 1 - nx1
        x1 = np.array([np.linspace(self.x_min, self.x_max, self.n)]).T  # preenche array com valores dentro do intervalo - linha - que é transposta
        X = np.column_stack([x0,x1])  #nxn  
        y = X @ self.w
        e = np.array([st.norm.rvs(size=self.n)]).T
        y = y + e
        return x1,y