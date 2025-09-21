import numpy as np
import abc
import src.optimizers as opt
import src.models as models 
import src.stop_criteria as stop
from src.preprocessing import Preprocessing

class Algorithm(abc.ABC):
    def __init__(self, optimizer_strategy: opt.OptimizerStrategy, model: models.Model) -> None:
        self.algorithm_observers = []
        self.optimizer_strategy = optimizer_strategy
        self.model = model
        
    def add(self, observer):
       if observer not in self.algorithm_observers:
           self.algorithm_observers.append(observer)
       else:
           print('Failed to add: {}'.format(observer))

    def remove(self, observer):
       try:
           self.algorithm_observers.remove(observer)
       except ValueError:
           print('Failed to remove: {}'.format(observer))

    def notify_iteration(self):
        [o.notify_iteration(self) for o in self.algorithm_observers]

    def notify_started(self):
        [o.notify_started(self) for o in self.algorithm_observers]

    def notify_finished(self):
        [o.notify_finished(self) for o in self.algorithm_observers]

    @abc.abstractmethod
    def fit(self, X, y, stop_criteria):
        """Implement the fit method"""

    @property
    def iteration(self):
        return self._iteration

    @iteration.setter
    def iteration(self, value):
        self._iteration = value    

    @property
    def errors(self):
        return self._errors
    
    @errors.setter
    def errors(self, value):
        self._errors = value

    @property
    def rmse(self):
        return self._rmse
    
    @rmse.setter
    def rmse(self, value):
        self._rmse = value
    
class PLA(Algorithm):
 
    def __init__(self, optimizer_strategy, model):
        super().__init__(optimizer_strategy, model)
 
    def fit(self, X, y, stop_criteria: stop.StopCriteria):
        self.iteration = 0
        (n, d) = X.shape
        X = np.column_stack([np.ones((n, 1)), X])
        self.model.w = np.zeros((d+1, 1))
        self.notify_started()
        while True:
            y_hat = self.model.predict(X)
            self.errors = y_hat - y
            w = self.model.w
            self.rmse = 1.0/n * (w.T @ X.T @ X @ w - 2*w.T @ X.T @ y + y.T @ y) # mse, nao rmse
            # self.rmse = 1/n * np.square(np.linalg.norm(y_hat - y))
            if stop_criteria.isFinished(self):
                self.notify_finished()
                break
            self.optimizer_strategy.update_model(X, y, self.model)
            self.iteration = self.iteration + 1
            self.notify_iteration()

