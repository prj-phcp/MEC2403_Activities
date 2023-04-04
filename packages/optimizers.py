import numpy as np

class GenericOptimizer:

    def __init__(self, tol, max_iter=200):
        
        self.tol = tol
        self.max_iter = max_iter
        self.iter = 0
        self.dims = 0
        self.clear_cache()

    def get_history(self):

        return self.cache_x
    
    def clear_cache(self):

        self.cache_x = []
        self.cache_grad = []

    def start_otim(self, x=None):

        self.iter = 0
        if x is None:
            self.ndims = 0
        else:
            self.dims = len(x)

    def get_direction(self, x, function):

        pass

    def __call__(self, function, p_initial, step):

        self.clear_cache()
        self.start_otim(p_initial)
        x = p_initial
        while self.iter < self.max_iter:
            self.cache_x.append(x)
            self.cache_grad.append(function.grad(*x))
            if np.linalg.norm(self.cache_grad[-1]) <= self.tol:
                return x
            direction = self.get_direction(x, function)
            _, x = step(x, direction, function)
            self.iter += 1
        return x
    
class UnivariantOptimizer(GenericOptimizer):

    def __init__(self, tol, max_iter=200):

        self.dirvectors = None
        super().__init__(tol, max_iter)

    def start_otim(self, x=None):

        super().start_otim(x)
        self.dirvectors = np.eye(self.dims)
        
    def get_direction(self, x, function):

        return self.dirvectors[self.iter % self.dims]
    

class PowellOptimizer(GenericOptimizer):

    def __init__(self, tol, max_iter=200):

        self.dirvectors = None
        super().__init__(tol, max_iter)

    def start_otim(self, x=None):

        super().start_otim(x)
        self.dirvectors = np.eye(self.dims)

    def get_direction(self, x, function):
        
        index = self.iter % (self.dims + 1)
        max_cycle = (self.dims + 1) ** 2
        cycle = self.iter % max_cycle
        if index == self.dims and self.iter > 0:
            d_vector = self.cache_x[-1] - self.cache_x[-3]
            self.dirvectors = np.vstack([self.dirvectors[1:,:], d_vector])
            return d_vector
        elif cycle == 0 and self.iter > 1:
            self.dirvectors = np.eye(self.dims)
        return self.dirvectors[index]
    

class SteepestDescentOptimizer(GenericOptimizer):

    def __init__(self, tol, max_iter=200):

        super().__init__(tol, max_iter)

    def get_direction(self, x, function):

        return -1*self.cache_grad[-1]
    

class FletcherReevesOptimizer(GenericOptimizer):

    def __init__(self, tol, max_iter=200):

        super().__init__(tol, max_iter)

    def clear_cache(self):

        self.cache_d = None
        super().clear_cache()

    def get_direction(self, x, function):
        grad_step = -1*self.cache_grad[-1]
        if self.iter == 0:
            self.cache_d = grad_step
        else:
            beta = (np.linalg.norm(self.cache_grad[-1])/np.linalg.norm(self.cache_grad[-2])) ** 2
            self.cache_d = grad_step + beta * self.cache_d
        return self.cache_d


class NewtonRaphsonOptimizer(GenericOptimizer):

    def __init__(self, tol, max_iter=200):

        super().__init__(tol, max_iter)

    def get_direction(self, x, function):
        
        return - np.linalg.solve(function.Hessian(*x), self.cache_grad[-1])


class BFGSOptimizer(GenericOptimizer):

    def __init__(self, tol, max_iter=200):

        self.S_matrix = None
        super().__init__(tol, max_iter)

    def clear_cache(self):

        self.cache_d = None
        super().clear_cache()

    def start_otim(self, x=None):

        super().start_otim(x)
        self.S_matrix = np.eye(self.dims)

    def update_S_matrix(self):

        if self.iter > 0:
            delta_x = self.cache_x[-1] - self.cache_x[-2]
            delta_grad = self.cache_grad[-1] - self.cache_grad[-2]

            # PAra deixar generico, ak foi calculado pelo produto escalar
            a_k = np.dot(self.cache_d, delta_x) / np.linalg.norm(self.cache_d)

            v_k = self.cache_grad[-2] * (1.0 + a_k * np.sqrt(
                np.abs(np.dot(self.cache_d, delta_grad) / np.dot(self.cache_grad[-2], delta_x)))) - self.cache_grad[-1]
            
            w_k = delta_x * (np.dot(delta_x, delta_grad))

            v_k, w_k = v_k.reshape(-1,1), w_k.reshape(-1,1)
            matrix = (np.eye(self.dims) + (w_k @ v_k.T))
            self.S_matrix = matrix @ self.S_matrix @ matrix.T
            print(matrix)

    def get_direction(self, x, function):

        grad_step = -1*self.cache_grad[-1]
        self.update_S_matrix()
        self.cache_d = np.matmul(self.S_matrix, grad_step)
        return self.cache_d
        

        