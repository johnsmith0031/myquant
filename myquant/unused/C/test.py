from myquant.garch import *

def test():
    ret = np.random.randn(100)*0.02
    garch = Garch(ret,0,0,1,1)
    garch.train()
    #print(garch.sigma)

def test2(name):
    print(name)
    
def test3(array):
    for i in range(len(array)):
        print(array[i])

def test4(ret):
    garch = Garch(ret,0,0,1,1)
    garch.train()
    return garch.sigma