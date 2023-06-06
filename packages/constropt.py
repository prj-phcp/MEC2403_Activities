from optimizers import GenericOptimizer
from steps import GenericStep
from constraints import ConstrainedSpecialFunction


class ConstrainedOptimizer:

    def __init__(self, tol, max_iter=200):

        self.tol = tol
        self.max_iter = max_iter
        self.clear_cache()

    def clear_cache(self):
        self.cache_x = []
        self.iter = 0

    def __call__(self, function:ConstrainedSpecialFunction, p_initial, optimizer:GenericOptimizer, step:GenericStep):

        self.clear_cache()
        function.start()
        x = p_initial
        while self.iter < self.max_iter:
            self.iter += 1
            self.cache_x.append(x)
            da_original = step.get_da()
            da = step.get_da()
            x_new = optimizer(function, x, step)
            while not function.check_validity(x_new):
                da = da/10.0
                step.set_da(da)
                x_new = optimizer(function, x, step)
            step.set_da(da_original)
            x = x_new
            if function.loss(*x) <= self.tol: 
                break
            function.step()
        self.cache_x.append(x)
        return x



