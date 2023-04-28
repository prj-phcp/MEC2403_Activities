import numpy as np

class ConstantStep:
    
    def __init__(self, da, epsilon=1e-8, check_direction=True, normalize=False, **kwargs):

        self.da = da
        self.epsilon = epsilon
        self.check_dir = check_direction
        self.aL = None
        self.aU = None
        self.fL = None
        self.fU = None
        self.multiplier = 1.0
        self.normalize = normalize
        self.reset_step()

    def reset_step(self):

        self.aL = 0
        self.aU = self.da
        self.fL =  0.0
        self.fU = -1.0
        self.multiplier = 1.0
        
    def calculate_deriv(self, p_initial, direction, function, aM):

        f_plus = function(*(p_initial + self.multiplier*(aM + self.epsilon)*direction))
        f_minus = function(*(p_initial + self.multiplier*(aM - self.epsilon)*direction))
        
        return f_plus - f_minus

    def calculate_bounds(self, p_initial, direction, function):

        self.fL = function(*(p_initial + self.multiplier*self.aL*direction))
        self.fU = function(*(p_initial + self.multiplier*self.aU*direction))

    def __call__(self, p_initial, direction, function):
        
        self.reset_step()
        if self.calculate_deriv(p_initial, direction, function, 0) > 0 and self.check_dir:
            self.multiplier = -1.0
        if self.normalize:
            self.multiplier = self.multiplier/np.linalg.norm(direction)
        self.calculate_bounds(p_initial, direction, function)
        while self.fL > self.fU:
            self.aL = self.aU
            self.aU += self.da
            self.calculate_bounds(p_initial, direction, function)
        pend = p_initial + self.multiplier*self.aL*direction
        
        return self.multiplier*self.aL, pend
    

class BissectionStep(ConstantStep):

    def __init__(self, da, tol, epsilon=1e-8, check_direction=True, normalize=False, **kwargs):

        if tol <= epsilon:
            self.tol = epsilon
        else:
            self.tol = tol

        super().__init__(da=da, epsilon=epsilon, check_direction=check_direction, normalize=normalize, **kwargs)
        

    def __call__(self, p_initial, direction, function):

        # Passo constante herdado da classe mae
        _, _ = super().__call__(p_initial, direction, function)

        self.aL -= self.da
        self.calculate_bounds(p_initial, direction, function)

        while self.aU - self.aL >= self.tol:
            aM = 0.5 * (self.aU + self.aL)
            deriv = self.calculate_deriv(p_initial, direction, function, aM)
            if deriv >= 0:
                self.aU = aM
            else:
                self.aL = aM

        ak = 0.5 * (self.aU + self.aL)
        deriv = self.calculate_deriv(p_initial, direction, function, ak)
        if deriv > 0:
            ak = self.aL
        elif deriv < 0:
            ak = self.aU
        pend = p_initial + self.multiplier*ak*direction

        return self.multiplier*ak, pend
    
class GoldenSectionStep(ConstantStep):

    def __init__(self, da, tol, epsilon=1e-8, check_direction=True, normalize=False, **kwargs):

        self.tol = tol
        self.ratio = 0.5*(np.sqrt(5) - 1.0)

        super().__init__(da=da, epsilon=epsilon, check_direction=check_direction, normalize=normalize, **kwargs)

        self.aE = None
        self.aD = None
        self.fE = None
        self.fD = None

        self.beta = None

    def calculate_D(self, p_initial, direction, function):

        self.aD = self.aL + self.ratio*self.beta
        self.fD = function(*(p_initial + self.multiplier*self.aD*direction))

    def calculate_E(self, p_initial, direction, function):

        self.aE = self.aL + (1-self.ratio)*self.beta
        self.fE = function(*(p_initial + self.multiplier*self.aE*direction))

    def __call__(self, p_initial, direction, function):

        # Passo constante herdado da classe mae
        _, _ = super().__call__(p_initial, direction, function)

        self.aL -= self.da
        self.calculate_bounds(p_initial, direction, function)

        # Iniciando as razoes aureas
        self.beta = self.aU - self.aL
        self.calculate_D(p_initial, direction, function)
        self.calculate_E(p_initial, direction, function)

        while self.beta >= self.tol:
            diff = self.fD - self.fE
            if diff >= 0:
                self.aU = self.aD
                self.fU = self.fD
                self.beta = self.aU - self.aL
                self.aD = self.aE
                self.fD = self.fE
                self.calculate_E(p_initial, direction, function)
            else:
                self.aL = self.aE
                self.fL = self.fE
                self.beta = self.aU - self.aL
                self.aE = self.aD
                self.fE = self.fD
                self.calculate_D(p_initial, direction, function)

        ak = 0.5 * (self.aU + self.aL)
        deriv = self.calculate_deriv(p_initial, direction, function, ak)
        if deriv > 0:
            ak = self.aL
        elif deriv < 0:
            ak = self.aU
        pend = p_initial + self.multiplier*ak*direction

        return self.multiplier*ak, pend
    

class AnalyticalStep:

    def __call__(self, p_initial, direction, function):
        
        #print(p_initial.shape)

        grad = function.grad(*p_initial).reshape(-1,1)
        direction = direction.reshape(-1,1)
        Q = function.Hessian(*p_initial)

        ak = np.dot(grad.T, direction) / (direction.T @ Q @ direction)
        ak = ak.reshape(-1)

        pend = pend = p_initial - ak*direction.reshape(-1)

        return ak, pend

