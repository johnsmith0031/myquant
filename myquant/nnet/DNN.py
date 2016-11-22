import numpy as np
import theano
import theano.tensor as T
from myquant.nnet.Layers import *

class DNN():
    
    def __init__(self, conv = [10,1,3,5,2], hidden = [20,10,1], core = 'relu', dropout = 0.2):
        
        layers = []
        layers.append(ConvLayer((100,conv[1],1,conv[0]),(conv[3],conv[1],1,conv[2]),(1,conv[4]),'conv',core))
        hidden[0] = (conv[0] - conv[2] + 1) // conv[4] * conv[3]
        self.unpack = hidden[0]
        for i in range(len(hidden)-2):
            layers.append(DenseLayer((100,hidden[i]),(hidden[i],hidden[i+1]),'layer'+str(i),core,dropout))
        layers.append(DenseLayer((100,hidden[-2]),(hidden[-2],hidden[-1]),'layer_end','linear',dropout))
        self.layers = layers
        
        self.params = []
        for layer in layers:
            self.params.extend(layer.params)
                    
          
    def output(self,input_x,input_y,learning_rate,dropout_flag):
                
        out = input_x
        out = self.layers[0].output(out,dropout_flag)
        out = out.flatten().reshape((-1,self.unpack))
        
        for layer in self.layers[1:]:
            out = layer.output(out,dropout_flag)
        
        loss = T.sum((out - input_y)**2)/2
        
        grads = T.grad(loss,self.params)
        
        updates = [
                (param_i, param_i - learning_rate * grad_i)\
                for param_i, grad_i in zip(self.params, grads)
                ]
                
        return out,loss,updates
        
    def get_train_function(self):
        input_x = T.tensor4('input_x')
        input_y = T.matrix('input_y')
        learning_rate = T.dscalar('learning_rate')
        dropout_flag = T.iscalar('dropout_flag')
        out, loss, updates = self.output(input_x,input_y,learning_rate,dropout_flag)
        func = theano.function([input_x,input_y,learning_rate,dropout_flag],(out,loss),updates = updates)
        self.func = func
    
    def train(self, x, y, epochs = 100, batch_size = 10, lr = 0.1, df = 0):
        hlen = len(x)
        for i in range(epochs):
            loss = 0
            for j in range(hlen//batch_size):
                out, loss_temp = self.func(x[(j*batch_size):((j+1)*batch_size)],y[(j*batch_size):((j+1)*batch_size)],lr,df)
                loss += loss_temp
            print('iter:'+str(i)+' loss:'+str(loss))
        print('train completed')
    
    def predict(self, x, ydim = 1):
        y = np.ones((len(x),ydim))
        yp,loss = self.func(x,y,0,0)
        return yp
    
if __name__ == '__main__':
    x = np.random.rand(100,4)
    y = x.sum(axis=1).reshape((-1,1))
    
    dnn = DNN([10,3,5,2],[20,10,1])
    dnn.get_train_function()
    dnn.train(xt.reshape((-1,1,1,10)),y*100,epochs = 1000,batch_size = 100,lr = 0.01)
    temp = dnn.predict(xt.reshape((-1,1,1,10)),1)
    temp = np.column_stack((y*100,temp))
    
    from myquant.functions_z1 import *
    
    linear(temp[:,0],temp[:,1])
    base = np.std(x,axis=0)
    xt = x/base
    
    linear(x,y)
    
    
    
    
    
    
    
    
    
    