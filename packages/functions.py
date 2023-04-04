import numpy as np
import numdifftools as nd
import copy

class SpecialFunction:

    def __init__(self):

        pass

    def __call__(self, *pargs):

        return 0.0
    
    def grad(self, *pargs):

        return np.zeros((len(pargs),))
    
    def Hessian(self, *pargs):

        return np.eye(len(pargs))
    
    def grad_mod(self, *pargs):

        return np.linalg.norm(self.grad(*pargs))
    
    def hess_sign(self, *pargs):

        n_comp = len(pargs)
        hess = self.Hessian(*pargs)
        eigenv = np.linalg.eigvals(hess)
        eigenpos = eigenv >= 0
        eigenneg = eigenv <= 0
        if np.sum(eigenpos) == n_comp:
            return 1
        elif np.sum(eigenneg) == n_comp:
            return -1
        else: 
            return 0

    

class AnalyticalSpecialFunction(SpecialFunction):

    def __init__(self, f, gradf, H):

        self.f = f
        self.H = H
        self.gradf = gradf

        super().__init__()

    def __call__(self, *pargs):

        return self.f(*pargs)

    def grad(self, *pargs):

        return self.gradf(*pargs)
    
    def Hessian(self, *pargs):

        return self.H(*pargs)
    

class NumericalSpecialFunction(AnalyticalSpecialFunction):

    def __init__(self, f, epsilon=1e-8):

        def f_vec(pargs):
            return f(*pargs)
        H = nd.Hessian(f_vec)
        gradf = nd.Gradient(f_vec)
        super().__init__(f, gradf, H)

    def grad(self, *pargs):

        return self.gradf(pargs)
    
    def Hessian(self, *pargs):

        return self.H(pargs)

    

    
