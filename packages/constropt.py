from optimizers import GenericOptimizer
from steps import GenericStep
from constraints import ConstrainedSpecialFunction


class ConstrainedOptimizer:

    def __init__(self, tol, max_iter=200, verbose=False):

        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.clear_cache()

    def clear_cache(self):
        self.cache_x = []
        self.iter = 0

    def __call__(self, function:ConstrainedSpecialFunction, p_initial, optimizer:GenericOptimizer, step:GenericStep):

        self.clear_cache()
        function.start()
        x = p_initial
        errorflag = False
        while self.iter < self.max_iter:
            self.iter += 1
            if self.verbose: print(f'    Beginning iteration: {self.iter}')
            self.cache_x.append(x)
            da_original = step.get_da()
            da = step.get_da()
            x_new = optimizer(function, x, step)
            while not function.check_validity(x_new):
                da = da/10.0
                if self.verbose: print(f'       Original delta alpha too big. trying delta alpha = {da}')
                if da <= self.tol:
                    errorflag = True
                    break
                step.set_da(da)
                x_new = optimizer(function, x, step)
            step.set_da(da_original)
            x = x_new
            if self.verbose: print(f'    Ending iteration: {self.iter}. Final point: {x[0]},{x[1]}, loss: {function.loss(*x)}')
            if function.loss(*x) <= self.tol: 
                break
            if errorflag:
                print('    A step value inferior to the tolerance is needed in order to solve the problem. The iteractive process stopped. The best solution is returned')
                break
            function.step()
        self.cache_x.append(x)
        return x



