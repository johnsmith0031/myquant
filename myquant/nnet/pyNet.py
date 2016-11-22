import numpy as np

class pyCNNLayer():
    
    def __init__(self,core_name = 'sigmoid'):
        
        self.core_name = core_name
        self.Weight = None
        self.bias = None
        self.max_down_size = None
    
    def conv2d(self,A,B):
        res = np.zeros((A.shape[0]-B.shape[0]+1,A.shape[1]-B.shape[1]+1))
        for i in range(A.shape[0]-B.shape[0]+1):
            for j in range(A.shape[1]-B.shape[1]+1):
                res[i,j] = np.sum(A[i:(i+B.shape[0]),j:(j+B.shape[1])]*B)
        return res

    def batch_conv(self,x,w):
        w = w[:,:,::-1,::-1]
        res = np.zeros((x.shape[0],w.shape[0],x.shape[2]-w.shape[2]+1,x.shape[3]-w.shape[3]+1))
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                for m in range(w.shape[0]):
                        res[i,m,:,:] += self.conv2d(x[i,j,:,:],w[m,j,:,:])
        return res
    
    def max_down(self,x,size):
        step = (int(x.shape[2]/size[0]),int(x.shape[3]/size[1]))
        res = np.zeros((x.shape[0],x.shape[1],step[0],step[1]))
        for i in range(step[0]):
            for j in range(step[1]):
                res[:,:,i,j] = np.max(x[:,:,(i*size[0]):((i+1)*size[0]),(j*size[1]):((j+1)*size[1])],axis = (2,3))
        return res
    
    def core(self,x):
        
        if self.core_name == 'sigmoid':
            res = 1/(1+np.exp(-x))
        elif self.core_name == 'softplus':
            res = np.log(np.exp(x)+1)
        elif self.core_name == 'relu':
            std = (np.std(x,axis=(1,2,3)) + 1)
            res = x*1
            for i in range(res.shape[0]):
                res[i,:,:,:] /= std[i]
            res[res<0] = 0
        elif self.core_name == 'abstanh':
            res = np.abs_(np.tanh(x))
        elif self.core_name == 'linear':
            res = x * 1.0
        elif self.core_name == 'one_relu':
            res = x*1
            res[res>1] = 1
            res[res<-1] = -1
        elif self.core_name == 'strelu':
            res = x*1
            res[res<0] = 0
        else:
            res = 1/(1+np.exp(-x))
        return res
    
    def output(self,input_x):
        
        out = self.batch_conv(input_x,self.Weight)
        for i in range(out.shape[1]):
            out[:,i,:,:] += self.bias[i]
        out = self.core(out)
        out = self.max_down(out,self.max_down_size)
        
        return out


class pyDenseLayer():
    
    def __init__(self,core_name = 'sigmoid'):
        
        self.core_name = core_name
        self.Weight = None
        self.bias = None
    
    def core(self,x):
        
        if self.core_name == 'sigmoid':
            res = 1/(1+np.exp(-x))
        elif self.core_name == 'softplus':
            res = np.log(np.exp(x)+1)
        elif self.core_name == 'relu':
            res = x / (np.std(x,axis=1).reshape((-1,1)) + 1)
            res[res<0] = 0
        elif self.core_name == 'abstanh':
            res = np.abs_(np.tanh(x))
        elif self.core_name == 'linear':
            res = x * 1.0
        elif self.core_name == 'one_relu':
            res = x*1
            res[res>1] = 1
            res[res<-1] = -1
        elif self.core_name == 'strelu':
            res = x*1
            res[res<0] = 0
        else:
            res = 1/(1+np.exp(-x))
        return res
    
    def output(self,input_x):
        
        out = self.core(input_x.dot(self.Weight) + self.bias)
        
        return out
        
class pyRNNLayer():
    
    def __init__(self,core_name = 'sigmoid'):
        
        self.core_name = core_name
        self.Weight = None
        self.Hidden_Weight = None
        self.bias = None
        self.hidden_unit = None
    
    def core(self,x):
        
        if self.core_name == 'sigmoid':
            res = 1/(1+np.exp(-x))
        elif self.core_name == 'softplus':
            res = np.log(np.exp(x)+1)
        elif self.core_name == 'relu':
            res = x / (np.std(x) + 1)
            res[res<0] = 0
        elif self.core_name == 'abstanh':
            res = np.abs_(np.tanh(x))
        elif self.core_name == 'linear':
            res = x * 1.0
        elif self.core_name == 'one_relu':
            res = x*1
            res[res>1] = 1
            res[res<-1] = -1
        elif self.core_name == 'strelu':
            res = x*1
            res[res<0] = 0
        else:
            res = 1/(1+np.exp(-x))
        return res
    
    def compute(self,input_x,hidden_unit):
        
        #in_x = np.concatenate((input_x,hidden_unit)).reshape((1,-1))
        x = input_x.dot(self.Weight) + hidden_unit.dot(self.Hidden_Weight) + self.bias
        out = self.core(x).flatten()
        
        return out
    
    def output(self,input_x):
        
        if self.hidden_unit is None:
            hidden_unit = np.zeros(self.Weight.shape[1])
        else:
            hidden_unit = self.hidden_unit*1
        res = np.zeros((input_x.shape[0],self.Weight.shape[1]))
        for i in range(input_x.shape[0]):
            res[i,:] = self.compute(input_x[i,:],hidden_unit)
            hidden_unit = res[i,:]
        self.hidden_unit = hidden_unit
        
        return res
        
class pySRNNLayer(pyRNNLayer):
    
    def __init__(self,core_name = 'sigmoid',depth = 5):
        
        self.core_name = core_name
        self.Weight = None
        self.Hidden_Weight = None
        self.bias = None
        self.depth = depth
    
    def init_online(self):
        self.former_input = []
    
    def compute_online(self,input_x):
        
        out = np.zeros(self.Weight.shape[1])
        hlen = len(self.former_input)
        for i in range(hlen):
            out = self.core(self.former_input[hlen-i-1].dot(self.Weight) + out.dot(self.Hidden_Weight) + self.bias)
            
        return out
        
    def online_output(self,input_x):
        
        self.former_input.insert(0,input_x)
        if len(self.former_input) > self.depth:
            self.former_input.pop()
        out = self.compute_online(input_x)
        
        return out
        
    def compute(self,input_x,hidden_unit,next_flag = True):
        
        x = input_x.dot(self.Weight) + hidden_unit.dot(self.Hidden_Weight) + self.bias
        out = self.core(x)
        if next_flag:
            out = np.concatenate((np.zeros((1,x.shape[1])),out[:-1]))
        
        return out
        
    def output_old(self,input_x):
        
        out = np.zeros((input_x.shape[0],self.Weight.shape[1]))
        for i in range(self.depth-1):
            out = self.compute(input_x,out)
        out = self.compute(input_x,out,False)
        
        return out
        
    def output(self,input_x):
        
        res = []
        for row in input_x:
            res.append(self.online_output(row))
        res = np.array(res)
        
        return res
        
#test = SRNNLayer((100,5),(5,1),'test','strelu',5)
#x = T.matrix('x')
#res = test.output(x)
#func = theano.function([x],res)
#
#xt = np.random.randn(100,5)
#yt = func(xt)
#
#test2 = pySRNNLayer('strelu',5)
#test2.Weight = test.Weight.get_value()
#test2.Hidden_Weight = test.Hidden_Weight.get_value()
#test2.bias = test.bias.get_value()
#test2.init_online()
#res = []
#for row in xt:
#    res.append(test2.online_output(row))
#res = np.array(res)
#yt2 = test2.output(xt)

class pyLSTMLayer():
    
    def __init__(self,core_name = 'tanh'):
        
        self.core_name = core_name
        self.Weight = None
        self.Weight_Forget = None
        self.bias_Forget = None
        self.Weight_Cell = None
        self.bias_Cell = None
        self.Weight_Update = None
        self.bias_Update = None
        self.Weight_Output = None
        self.bias_Output = None
    
    def core(self,x,core_name):
        
        if core_name == 'sigmoid':
            res = 1/(1+np.exp(-x))
        elif core_name == 'softplus':
            res = np.log(np.exp(x)+1)
        elif core_name == 'relu':
            res = x / (np.std(x,axis=0).reshape((-1,1)) + 1)
            res[res<0] = 0
        elif core_name == 'abstanh':
            res = np.abs_(np.tanh(x))
        elif core_name == 'tanh':
            res = np.tanh(x)
        elif core_name == 'linear':
            res = x * 1.0
        elif core_name == 'one_relu':
            res = x*1
            res[res>1] = 1
            res[res<-1] = -1
        elif self.core_name == 'strelu':
            res = x*1
            res[res<0] = 0
        else:
            res = 1/(1+np.exp(-x))
        return res
    
    def compute(self,input_x,hidden_unit,cell_unit):
        
        in_x = input_x.dot(self.Weight)
        in_x = np.concatenate((in_x.flatten(),hidden_unit.flatten())).reshape((2,-1))
        forget_gate = self.core(self.Weight_Forget.dot(in_x) + self.bias_Forget,'sigmoid').flatten()
        update_gate = self.core(self.Weight_Update.dot(in_x) + self.bias_Update,'sigmoid').flatten()
        cell_content = self.core(self.Weight_Cell.dot(in_x) + self.bias_Cell,self.core_name).flatten()
        output_gate = self.core(self.Weight_Output.dot(in_x) + self.bias_Output,'sigmoid').flatten()
        cell_unit  = cell_unit*forget_gate + cell_content*update_gate
        hidden_unit = self.core(cell_unit,self.core_name)*output_gate
        
        return hidden_unit,cell_unit
    
    def output(self,input_x):
        
        hidden_unit = np.zeros(self.Weight.shape[1])
        cell_unit = np.zeros(self.Weight.shape[1])
        res = np.zeros((input_x.shape[0],self.Weight.shape[1]))
        for i in range(input_x.shape[0]):
            hidden_unit,cell_unit = self.compute(input_x[i,:],hidden_unit,cell_unit)
            res[i,:] = hidden_unit
            
        return res

    def init_pars(self):
        
        self.Weight = np.random.randn(10,5)
        self.Weight_Forget = np.random.randn(1,2)
        self.bias_Forget = np.random.randn(1)
        self.Weight_Cell = np.random.randn(1,2)
        self.bias_Cell = np.random.randn(1)
        self.Weight_Update = np.random.randn(1,2)
        self.bias_Update = np.random.randn(1)
        self.Weight_Output = np.random.randn(1,2)
        self.bias_Output = np.random.randn(1)
        

class Net():
    
    def __init__(self,hidden = ['relu','relu','sigmoid'],layer_type = None):
        
        if layer_type is None:
            layer_type = []
            for i in range(len(hidden)):
                layer_type.append('Dense')
                
        if len(hidden) != len(layer_type):
            raise Exception('length mismatch')
            
        types = []
        for i in layer_type:
            if i == 'RNN':
                types.append(pyRNNLayer)
            elif i == 'LSTM':
                types.append(pyLSTMLayer)
            elif i == 'CNN':
                types.append(pyCNNLayer)
            elif i == 'SRNN':
                types.append(pySRNNLayer)
            else:
                types.append(pyDenseLayer)
        
        layers = []
        for i in range(len(hidden)):
            layers.append(types[i](hidden[i]))
        self.layers = layers
        self.layer_type = layer_type
                            
    def set_params(self,pars,shapes):
        re_cons = []
        loc = 0
        for row in shapes:
            hlen = np.prod(row[row!=0])
            re_cons.append(pars[int(loc):int(loc+hlen)].reshape(np.array(row[row!=0],dtype = int)))
            loc += hlen
        
        loc = 0
        for i in range(len(self.layers)):
            if self.layer_type[i] == 'RNN' or self.layer_type[i] == 'SRNN':
                self.layers[i].Weight = re_cons[loc]
                self.layers[i].Hidden_Weight = re_cons[loc+1]
                self.layers[i].bias = re_cons[loc+2]
                loc += 3
            elif self.layer_type[i] == 'LSTM':
                self.layers[i].Weight = re_cons[loc]
                self.layers[i].Weight_Forget = re_cons[loc+1]
                self.layers[i].bias_Forget = re_cons[loc+2]
                self.layers[i].Weight_Cell = re_cons[loc+3]
                self.layers[i].bias_Cell = re_cons[loc+4]
                self.layers[i].Weight_Update = re_cons[loc+5]
                self.layers[i].bias_Update = re_cons[loc+6]
                self.layers[i].Weight_Output = re_cons[loc+7]
                self.layers[i].bias_Output = re_cons[loc+8]
                loc += 9
            else:
                self.layers[i].Weight = re_cons[loc]
                self.layers[i].bias = re_cons[loc+1]
                loc += 2
                
    def load_params_from_file(self,par_path = '',shape_path = ''):
        temp = np.loadtxt(par_path)
        temp2 = np.loadtxt(shape_path)
        pars = temp
        shapes = np.array(temp2.reshape((-1,4)),dtype=int)
        return pars,shapes
        
    def load_params(self,par_path = '',shape_path = ''):
        pars,shapes = self.load_params_from_file(par_path,shape_path)
        self.set_params(pars,shapes)
    
    def read_params(self,par_path = '',shape_path = ''):
        temp = np.loadtxt(par_path)
        temp2 = np.loadtxt(shape_path)
        pars = temp
        shapes = np.array(temp2.reshape((-1,4)),dtype=int)
        return pars,shapes
    
    def output(self,input_x):
                  
        out = input_x
        
        cnn_flag = False
        for layer in self.layers:
            if cnn_flag and type(layer) != pyCNNLayer:
                out = out.flatten().reshape((-1,layer.Weight.shape[0]))
            out = layer.output(out)
            if type(layer) == pyCNNLayer:
                cnn_flag = True
            else:
                cnn_flag = False
        
        return out
        