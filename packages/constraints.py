from functions import SpecialFunction
import numpy as np
from typing import List


class Constraint(SpecialFunction):

    def __init__(self, func : SpecialFunction, r0 : float, beta : float):

        self.func = func
        self.r0 = r0
        self.beta = beta
        self.reset_r()

    def reset_r(self):

        self.rp = self.r0

    def step(self):

        self.rp *= self.beta

    def check_validity(self, point):

        return True


class PenaltyEquityConstraint(Constraint):

    
    def __call__(self, *pargs):

        return 0.5 * self.rp * (self.func(*pargs)) ** 2
    
    def grad(self, *pargs):

        return self.rp * self.func(*pargs) * self.func.grad(*pargs)
    
    def Hessian(self, *pargs):
        gradf = self.func.grad(*pargs).reshape(-1,1)
        return self.rp * (self.func(*pargs) * self.func.Hessian(*pargs) + gradf.T @ gradf)
    
    

class PenaltyInequityConstraint(Constraint):


    def __call__(self, *pargs):

        return 0.5 * self.rp * (np.max([0,self.func(*pargs)])) ** 2
    
    def grad(self, *pargs):

        return self.rp * np.max([0.0,self.func(*pargs)]) * self.func.grad(*pargs)
    
    def Hessian(self, *pargs):
        f = self.func(*pargs)
        gradf = self.func.grad(*pargs).reshape(-1,1)
        return self.rp * (0.5 * (1.0 + np.sign(f))) * (f * self.func.Hessian(*pargs) + gradf.T @ gradf)
    

class BarrierInequityConstraint(Constraint):

    
    def __call__(self, *pargs):

        return -1.0 * self.rp / (self.func(*pargs))
    
    def grad(self, *pargs):

        return self.rp / (self.func(*pargs)) ** 2 * self.func.grad(*pargs)
    
    def Hessian(self, *pargs):
        f = self.func(*pargs)
        gradf = self.func.grad(*pargs).reshape(-1,1)
        return self.rp / (f) ** 2 * (self.func.Hessian(*pargs) - 2.0 / f *  gradf.T @ gradf)
    
    def check_validity(self, point):

        return self.func(*point) <= 0.0
    

class ConstrainedSpecialFunction(SpecialFunction):

    def __init__(self, func:SpecialFunction, hfuncs:List[Constraint]=[], cfuncs:List[Constraint]=[]):

        self.func = func
        self.hfuncs = hfuncs
        self.cfuncs = cfuncs
        self.all_constraints = self.hfuncs + self.cfuncs
    
    def start(self):

        for constraint in self.all_constraints:
            constraint.reset_r()

    def step(self):

        for constraint in self.all_constraints:
            constraint.step()

    def __call__(self, *pargs):

        f = self.func(*pargs)
        for constraint in self.all_constraints:
            f += constraint(*pargs)
        return f
    
    def grad(self, *pargs):

        gradf = self.func.grad(*pargs)
        for constraint in self.all_constraints:
            gradf += constraint.grad(*pargs)
        return gradf

    def Hessian(self, *pargs):

        hessf = self.func.Hessian(*pargs)
        for constraint in self.all_constraints:
            hessf += constraint.Hessian(*pargs)
        return hessf
    
    def loss(self, *pargs):

        loss = 0.0
        for constraint in self.all_constraints:
            loss += constraint(*pargs)
        return loss
    
    def check_validity(self, point):

        validity = True
        for constraint in self.all_constraints:
            validity = validity and constraint.check_validity(point)
        return validity

