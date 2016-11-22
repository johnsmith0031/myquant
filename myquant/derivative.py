import numpy as np

def get_jacobian(f,x,ld = 0.0001,loc = -1):
    
    J = np.zeros_like(x)
    eps = ld * x
    for i in range(len(x)):
        temp_x1 = x*1
        temp_x1[i] += eps[i]
        temp_x2 = x*1
        temp_x2[i] -= eps[i]
        if loc != -1:
            J[i] = ((f(temp_x1) - f(temp_x2))/(2*eps[i]))[loc]
        else:
            J[i] = (f(temp_x1) - f(temp_x2))/(2*eps[i])
            
    return J
        
def get_hessian(f,x,ld = 0.0001):  
    
    hlen = len(x)
    eps = ld * x
    H = np.zeros((hlen,hlen))
    for i in range(hlen):
        for j in range(hlen):
            x1,x2,x3,x4 = x*1,x*1,x*1,x*1
            x1[i] += eps[i]
            x1[j] += eps[j]
            x2[i] += eps[i]
            x2[j] -= eps[j]
            x3[i] -= eps[i]
            x3[j] += eps[j]
            x4[i] -= eps[i]
            x4[j] -= eps[j]
            H[i,j] = (f(x1) - f(x2) - f(x3) + f(x4))/(4*eps[i]*eps[j])
            
    return H