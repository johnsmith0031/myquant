import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from theano.tensor.shared_randomstreams import RandomStreams
import copy
from decimal import Decimal,getcontext
import io

try:
    from pylearn2.expr.stochastic_pool import *
except:
    print('Warn: pylearn2 is not installed, ConvLayer is not available')
    
theano.config.floatX = 'float64'

class BaseLayer():
    
    def core(self,x):
        
        if self.core_name == 'sigmoid':
            res = T.nnet.sigmoid(x)
        elif self.core_name == 'softplus':
            res = T.log(T.exp(x)+1)
        elif self.core_name == 'relu':
            res = x / (T.std(x,axis=1).reshape((-1,1)) + 1)
            res = T.switch(res > 0, res, 0)
        elif self.core_name == 'abstanh':
            res = T.abs_(T.tanh(x))
        elif self.core_name == 'tanh':
            res = T.tanh(x)
        elif self.core_name == 'linear':
            res = x * 1.0
        elif self.core_name == 'one_relu':
            res = T.switch(x < 1, x, 1)
            res = T.switch(res > -1, res, -1)
        elif self.core_name == 'strelu':
            res = x
            res = T.switch(res > 0, res, 0)
        else:
            res = T.nnet.sigmoid(x)
            
        return res

def save_to_file(path,data,accuracy = 50):
    getcontext().prec = accuracy
    string = ''
    for i in data:
        string += str(Decimal(i)) + '    '
        
    try:
        file = io.open(path,'w')
        file.write(string)
        file.close()
        print('data saved')
    except Exception as e:
        print(e)
        file.close()
        
def compile_func(CLASS,conv_flag = False):
    if conv_flag:
        input_x = T.tensor4('input_x')
    else:
        input_x = T.matrix('input_x')
    input_y = T.matrix('input_y')
    learning_rate = T.dscalar('learning_rate')
    flag = T.dscalar('flag')
    rnn = CLASS()
    out,loss,updates = rnn.output(input_x,input_y,learning_rate,flag)
    func = theano.function([input_x,input_y,learning_rate,flag],(out,loss),updates = updates,on_unused_input='ignore')
    return rnn,func

class Pretrain_Wrapper():
    
    def __init__(self,nn):
        self.nn = nn
        self.init_layer_adam()
        self.init_bound()
        
    def init_layer_adam(self):
        self.b1,self.b2,self.ep = 0.9,0.999,1e-8
        for layer in self.nn.layers:
            assert type(layer) in [DenseLayer,SRNNLayer]
            layer.ms = []
            layer.vs = []
            for param in layer.pretrain_params:
                ele = copy.deepcopy(param)
                ele.name += '_m_pre'
                ele.set_value(np.zeros_like(ele.get_value()))
                layer.ms.append(ele)
                ele = copy.deepcopy(param)
                ele.name += '_v_pre'
                ele.set_value(np.zeros_like(ele.get_value()))
                layer.vs.append(ele)

    def init_bound(self):
        self.bound = 1e10
        self.chushu = 2
    
    def get_output_loss_by_layer(self,input_x,k):
        
        hid,vis = self.nn.layers[k].reconstruct(input_x)
        loss = T.sum((vis-input_x)**2)/2
        
        return hid,vis,loss
        
    def get_grad_by_layer(self,loss,k):
        
        params = self.nn.layers[k].pretrain_params
        grads = T.grad(loss,params)
        
        return grads
        
    def adam_decsent_by_layer(self,grads,learning_rate,k):
        
        ms_updated = [self.b1*m + (1-self.b1)*grad for m,grad in zip(self.nn.layers[k].ms,grads)]
        vs_updated = [self.b2*v + (1-self.b2)*grad**2 for v,grad in zip(self.nn.layers[k].vs,grads)]
        ms_hat = ms_updated
        vs_hat = vs_updated
        
        updates = []
        for param, m_hat,v_hat in zip(self.nn.layers[k].pretrain_params, ms_hat,vs_hat):
            p_update = param - learning_rate*m_hat/(T.sqrt(v_hat)+0.0001)
            p_update = T.switch(T.abs_(p_update)>self.bound,p_update/self.chushu,p_update)
            updates.append((param,p_update))
            
        updates.extend([(m,m_updated) for m,m_updated in zip(self.nn.layers[k].ms,ms_updated)])
        updates.extend([(v,v_updated) for v,v_updated in zip(self.nn.layers[k].vs,vs_updated)])
        
        return updates
        
    def get_updates_by_layer(self,input_x,learning_rate,k):
        
        hid,vis,loss = self.get_output_loss_by_layer(input_x,k)
        grads = self.get_grad_by_layer(loss,k)
        updates = self.adam_decsent_by_layer(grads,learning_rate,k)
        
        return hid,vis,loss,updates
        
    def compile_func_by_layer(self,k):
        
        input_x = T.matrix('input_x')
        learning_rate = T.dscalar('learning_rate')
        hid,vis,loss,updates = self.get_updates_by_layer(input_x,learning_rate,k)
        func = theano.function([input_x,learning_rate],(hid,vis,loss),updates = updates)
        
        return func
        
    def compile_funcs(self):
        
        funcs = []
        for i in range(len(self.nn.layers)):
            funcs.append(self.compile_func_by_layer(i))
            print('Layer ' + str(i+1) + ' Training Function Compiled')
            
        return funcs
        
    def pretrain(self,funcs,input_x,max_iter = 10000):
        loc = 0
        hid_in = input_x
        for func in funcs:
            old_loss = 0
            for i in range(max_iter):
                hid,vis,loss = func(hid_in,0.01)
                if (old_loss - loss) <= 1e-4 and old_loss != 0:
                    break
                old_loss = loss
                if i > 0 and i%1000 == 0:
                    print('Epoch ' + str(i) + ' Loss:' + str(loss))
            hid_in = hid
            loc += 1
            print('Layer ' + str(loc) + ' Trained. ' + str(i) + ' Epoch Took. Loss: ' + str(loss))
            
class NN():
    
    def __init__(self):
        self.set_layers()
        self.init_pars_list()
    
    def set_layers(self):
        layers = []
        layers.append(DenseLayer((100,5),(5,3),'layer1','relu',0))
        layers.append(RNNLayer((100,3),(3,1),'layer2','linear'))
        self.layers = layers
    
    def clear_pars(self):
        self.load_pars(self.init_pars,self.init_shapes)
    
    def save_pars(self,path,pars,shapes):
        
        shapes = shapes.flatten()
        if path[-2:] == '\\':
            save_to_file(path + 'pars.txt',pars)
            save_to_file(path + 'shapes.txt',shapes)
        else:
            save_to_file(path + '\\pars.txt',pars)
            save_to_file(path + '\\shapes.txt',shapes)
        
    def get_pars(self):
        
        #get parameters and save
        param_record = []
        for param in self.params:
            param_record.append(param.get_value())
            
        #change_list_params
        shapes = np.zeros((len(param_record),4))
        pars = np.array([])
        loci = 0
        for ele in param_record:
            locj = 0
            for k in ele.shape:
                shapes[loci,locj] = k
                locj += 1
            pars = np.concatenate((pars,ele.flatten()))
            loci += 1
            
        return pars,shapes
        
    def init_pars_list(self):
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)
            
        self.init_pars,self.init_shapes = self.get_pars()
    
    def load_pars(self,pars,shapes):
        re_cons = []
        loc = 0
        for row in shapes:
            hlen = np.prod(row[row!=0])
            re_cons.append(pars[int(loc):int(loc+hlen)].reshape(np.array(row[row!=0],dtype = int)))
            loc += hlen
        
        #load_network_params
        for i in range(len(self.params)):
            self.params[i].set_value(re_cons[i])
    
    def get_loss(self,input_x,input_y,learning_rate,flag):
        
        out = input_x
        
        cnn_flag = False
        for layer in self.layers:
            if cnn_flag and type(layer) != ConvLayer:
                    out = out.flatten().reshape((-1,layer.Weight_shape[0]))
            if type(layer) == DenseLayer or type(layer) == ConvLayer:
                out = layer.output(out,flag)
            else:
                out = layer.output(out)
            if type(layer) == ConvLayer:
                cnn_flag = True
            else:
                cnn_flag = False
        
        loss = self.lossfunc(out,input_y)
        
        return out,loss
        
    def lossfunc(self,out,input_y):
        
        loss = T.sum((out.flatten()-input_y.flatten())**2)/2
        
        return loss
    
    def output(self,input_x,input_y,learning_rate,flag):
                  
        out,loss = self.get_loss(input_x,input_y,learning_rate,flag)
        
        grads = T.grad(loss,self.params)
        
        updates = [
                (param_i, param_i - learning_rate * grad_i)\
                for param_i, grad_i in zip(self.params, grads)
                ]

        return out,loss,updates


class ADAMNN(NN):
    
    def __init__(self):
        self.set_layers()
        self.init_pars_list()
        self.init_adam()
        self.init_grad_wapper()
        self.init_bound()
        
    def init_adam(self):
        self.b1,self.b2,self.ep = 0.9,0.999,1e-8
        self.ms = []
        self.vs = []
        for param in self.params:
            ele = copy.deepcopy(param)
            ele.name += '_m'
            ele.set_value(np.zeros_like(ele.get_value()))
            self.ms.append(ele)
            ele = copy.deepcopy(param)
            ele.name += '_v'
            ele.set_value(np.zeros_like(ele.get_value()))
            self.vs.append(ele)

    def init_bound(self):
        self.bound = 1e10
        self.chushu = 2
            
    def init_grad_wapper(self):
        self.grads = []
        for param in self.params:
            ele = copy.deepcopy(param)
            ele.name += '_grad'
            ele.set_value(np.zeros_like(ele.get_value()))
            self.grads.append(ele)
            
    def clear_pars(self):
        for m,v,grad in zip(self.ms,self.vs,self.grads):
            m.set_value(np.zeros_like(m.get_value()))
            v.set_value(np.zeros_like(v.get_value()))
            grad.set_value(np.zeros_like(grad.get_value()))
        self.load_pars(self.init_pars,self.init_shapes)
    
    def compute_grad(self,loss):
        
        grads = T.grad(loss,self.params)
        
        return grads
    
    def adam_decsent(self,grads,learning_rate):
                
        ms_updated = [self.b1*m + (1-self.b1)*grad for m,grad in zip(self.ms,grads)]
        vs_updated = [self.b2*v + (1-self.b2)*grad**2 for v,grad in zip(self.vs,grads)]
        ms_hat = ms_updated
        vs_hat = vs_updated
        
        updates = []
        for param, m_hat,v_hat in zip(self.params, ms_hat,vs_hat):
            p_update = param - learning_rate*m_hat/(T.sqrt(v_hat)+0.0001)
            p_update = T.switch(T.abs_(p_update)>self.bound,p_update/self.chushu,p_update)
            updates.append((param,p_update))
            
        updates.extend([(m,m_updated) for m,m_updated in zip(self.ms,ms_updated)])
        updates.extend([(v,v_updated) for v,v_updated in zip(self.vs,vs_updated)])
        updates.extend([(g,grad) for g,grad in zip(self.grads,grads)])
        
        return updates
    
    def output(self,input_x,input_y,learning_rate,flag):
        
        out,loss = self.get_loss(input_x,input_y,learning_rate,flag)
        grads = self.compute_grad(loss)
        updates = self.adam_decsent(grads,learning_rate)
        
        return out,loss,updates


class LSTMLayer(BaseLayer):
    
    def __init__(self,input_shape,Weight_shape,layername = '',core_name = 'sigmoid'):
        
        self.rng = np.random.RandomState(2500)
        self.input_shape = input_shape
        self.Weight_shape = Weight_shape
        self.Weight = theano.shared(np.asarray(self.rng.uniform(-1,1,size = self.Weight_shape),
                                               dtype = theano.config.floatX),name = layername+'_Weight')
        self.Weight_Forget = theano.shared(np.asarray(self.rng.uniform(-1,1,size = (1,2)),
                                               dtype = theano.config.floatX),name = layername+'_Weight_Forget')
        self.bias_Forget = theano.shared(np.asarray(self.rng.uniform(-0,0,size = (1)),
                                             dtype = theano.config.floatX),name = layername+'_bias_Forget')
        self.Weight_Cell = theano.shared(np.asarray(self.rng.uniform(-1,1,size = (1,2)),
                                               dtype = theano.config.floatX),name = layername+'_Weight_Cell')
        self.bias_Cell = theano.shared(np.asarray(self.rng.uniform(-0,0,size = (1)),
                                             dtype = theano.config.floatX),name = layername+'_bias_Cell')
        self.Weight_Update = theano.shared(np.asarray(self.rng.uniform(-1,1,size = (1,2)),
                                               dtype = theano.config.floatX),name = layername+'_Weight_Update')
        self.bias_Update = theano.shared(np.asarray(self.rng.uniform(-0,0,size = (1)),
                                             dtype = theano.config.floatX),name = layername+'_bias_Update')
        self.Weight_Output = theano.shared(np.asarray(self.rng.uniform(-1,1,size = (1,2)),
                                               dtype = theano.config.floatX),name = layername+'_Weight_Output')
        self.bias_Output = theano.shared(np.asarray(self.rng.uniform(-0,0,size = (1)),
                                             dtype = theano.config.floatX),name = layername+'_bias_Output')
        self.layername = layername
        self.core_name = core_name
        self.hidden_unit = theano.shared(np.asarray(np.zeros(self.Weight_shape[1]),dtype = theano.config.floatX),name = layername+'hidden_unit')
        self.cell_unit = theano.shared(np.asarray(np.zeros(self.Weight_shape[1]),dtype = theano.config.floatX),name = layername+'cell_unit')

        self.params = [self.Weight,self.Weight_Forget,self.bias_Forget,
                       self.Weight_Cell,self.bias_Cell,self.Weight_Update,self.bias_Update,
                       self.Weight_Output,self.bias_Output]
                            
    def compute(self,input_x,hidden_unit,cell_unit):
        
        in_x = input_x.dot(self.Weight)
        in_x = T.concatenate((in_x,hidden_unit)).reshape((2,-1))
        forget_gate = self.core(self.Weight_Forget.dot(in_x) + self.bias_Forget.dimshuffle(0,'x'),'sigmoid').flatten()
        update_gate = self.core(self.Weight_Update.dot(in_x) + self.bias_Update.dimshuffle(0,'x'),'sigmoid').flatten()
        cell_content = self.core(self.Weight_Cell.dot(in_x) + self.bias_Cell.dimshuffle(0,'x'),self.core_name).flatten()
        output_gate = self.core(self.Weight_Output.dot(in_x) + self.bias_Output.dimshuffle(0,'x'),'sigmoid').flatten()
        cell_unit  = cell_unit*forget_gate + cell_content*update_gate
        hidden_unit = self.core(cell_unit,self.core_name)*output_gate
        
        return hidden_unit,cell_unit
        
    def output(self,input_x):
        
        self.hidden_unit.set_value(np.zeros(self.Weight_shape[1]))
        self.cell_unit.set_value(np.zeros(self.Weight_shape[1]))
        result, update = theano.scan(fn = self.compute,
                                     outputs_info = [dict(initial=self.hidden_unit, taps = [-1]),dict(initial=self.cell_unit, taps = [-1])],
                                     sequences = input_x,
                                     truncate_gradient = 10)
        
        return result[0]

class RNNLayer(BaseLayer):
    
    def __init__(self,input_shape,Weight_shape,layername = '',core_name = 'sigmoid'):
        
        self.init_all(input_shape,Weight_shape,layername,core_name)
        
    def init_all(self,input_shape,Weight_shape,layername,core_name):
        #Weight_shape = (Weight_shape[0] + Weight_shape[1],Weight_shape[1])
        self.rng = np.random.RandomState(2500)
        self.input_shape = input_shape
        self.Weight_shape = Weight_shape
        self.Weight = theano.shared(np.asarray(self.rng.uniform(-1,1,size = self.Weight_shape),
                                               dtype = theano.config.floatX),name = layername+'_Weight')
        self.Hidden_Weight = theano.shared(np.asarray(self.rng.uniform(-1,1,size = (self.Weight_shape[1],self.Weight_shape[1])),
                                               dtype = theano.config.floatX),name = layername+'_Hidden_Weight')
        self.bias = theano.shared(np.asarray(self.rng.uniform(-0,0,size = self.Weight_shape[1]),
                                             dtype = theano.config.floatX),name = layername+'_bias')
        self.layername = layername
        self.core_name = core_name
        self.init_value = theano.shared(np.asarray(np.zeros(self.Weight_shape[1]),dtype = theano.config.floatX),name = 'init_value')
        
        self.params = [self.Weight,self.Hidden_Weight,self.bias]
                    
    
    def compute(self,input_x,hidden_unit):
                                    
        #in_x = T.concatenate((input_x,hidden_unit))
        input_x = input_x.reshape((1,-1))
        hidden_unit = hidden_unit.reshape((1,-1))
        out = self.core(input_x.dot(self.Weight) + hidden_unit.dot(self.Hidden_Weight) + self.bias.dimshuffle('x',0)).flatten()
        
        return out
        
    def output(self,input_x):
        
        self.init_value.set_value(np.zeros(self.Weight_shape[1]))
        result, update = theano.scan(fn = self.compute,
                                     outputs_info = dict(initial = self.init_value,taps = [-1]),
                                     sequences = input_x,
                                     truncate_gradient = 10)
        
        return result

class SRNNLayer(RNNLayer):
    
    def __init__(self,input_shape,Weight_shape,layername = '',core_name = 'sigmoid',depth = 5):
        
        self.depth = depth
        self.init_all(input_shape,Weight_shape,layername,core_name)
        self.bias_vis = theano.shared(np.asarray(self.rng.uniform(-0,0,size = self.Weight_shape[0]),
                                                 dtype = theano.config.floatX),name = layername+'_bias_vis')
        self.pretrain_params = [self.Weight,self.bias,self.bias_vis]
        
    def compute(self,input_x,hidden_unit,next_flag = True):
        
        res = self.core(input_x.dot(self.Weight) + hidden_unit.dot(self.Hidden_Weight) + self.bias.dimshuffle('x',0))
        if next_flag:
            res = T.concatenate((T.zeros((1,res.shape[1]))*1.0,res[:-1]),axis = 0)
        
        return res
    
    def output(self,input_x):
        
        out = self.init_value
        for i in range(self.depth-1):
            out = self.compute(input_x,out)
        out = self.compute(input_x,out,False)
        
        return out
        
    def reconstruct(self,input_x):
        
        in_x = input_x
        hid = self.core(in_x.dot(self.Weight) + self.bias.dimshuffle('x',0))
        vis = self.core(hid.dot(self.Weight.T) + self.bias_vis.dimshuffle('x',0))
        
        return hid,vis
        
#x = T.matrix('x')
#y = T.matrix('x')
#test = SRNNLayer((100,5),(5,1),'test','strelu',5)
#res = test.compute(x,y)
#func = theano.function([x,y],res)
#
#test.Weight.set_value(np.ones((5,1)))
#test.Hidden_Weight.set_value(np.ones((1,1)))
#xt = np.ones((100,5))
#res = np.zeros((100,1))
#res = func(xt,res)

class ConvLayer():
    
    def __init__(self,input_shape,filter_shape,max_pool_size,layername = '',core_name = 'sigmoid'):
        
        self.rng = np.random.RandomState(2500)
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.max_pool_size = max_pool_size
        self.filter_W = theano.shared(np.asarray(self.rng.uniform(-1,1,size = self.filter_shape),
                                                 dtype = theano.config.floatX),name = layername+'filter_W')
        self.filter_b = theano.shared(np.asarray(self.rng.uniform(0,0,size = self.filter_shape[0]),
                                                 dtype = theano.config.floatX),name = layername+'filter_W')
        self.layername = layername
        self.core_name = core_name
        
        self.params = [self.filter_W, self.filter_b]
        
    def core(self,x):
        if self.core_name == 'sigmoid':
            res = T.nnet.sigmoid(x)
        elif self.core_name == 'softplus':
            res = T.log(T.exp(x)+1)
        elif self.core_name == 'relu':
            #res = x / T.switch(T.eq(T.std(x),0), 1, T.std(x))
            res = x / (T.std(x,axis=(1,2,3)).dimshuffle((0,'x','x','x')) + 1)
            res = T.switch(res > 0, res, 0)
        elif self.core_name == 'abstanh':
            res = T.abs_(T.tanh(x))
        elif self.core_name == 'tanh':
            res = T.tanh(x)
        elif self.core_name == 'linear':
            res = x
        elif self.core_name == 'one_relu':
            res = T.switch(x < 1, x, 1)
            res = T.switch(res > -1, res, -1)
        elif self.core_name == 'strelu':
            res = x
            res = T.switch(res > 0, res, 0)
        else:
            res = T.nnet.sigmoid(x)
        
        return res
        
    def output(self,input_x,sto):
        
        con = conv.conv2d(input_x,self.filter_W)
        shape = (self.input_shape[-2] - self.filter_shape[-2] + 1,
                 self.input_shape[-1] - self.filter_shape[-1] + 1)
        don = T.switch(T.eq(sto,1),
                       weighted_max_pool_bc01(con,self.max_pool_size,self.max_pool_size,shape),
                       downsample.max_pool_2d(con,self.max_pool_size,ignore_border = False))
        don = self.core(don + self.filter_b.dimshuffle('x', 0, 'x', 'x'))
        
        return don
        
class SoftMaxLayer():
    
    def __init__(self):
        
        self.params = []
                        
    def output(self,input_x):
        
        out = (input_x) / T.sum(input_x,axis=1).reshape((-1,1))
        
        return out

class DenseLayer(BaseLayer):
    
    def __init__(self,input_shape,Weight_shape,layername = '',core_name = 'sigmoid',dropout = 0):
        
        self.rng = np.random.RandomState(2500)
        self.rng2 = RandomStreams(3000)
        self.input_shape = input_shape
        self.Weight_shape = Weight_shape
        self.Weight = theano.shared(np.asarray(self.rng.uniform(-1,1,size = self.Weight_shape),
                                               dtype = theano.config.floatX),name = layername+'_Weight')
        self.bias = theano.shared(np.asarray(self.rng.uniform(-0,0,size = self.Weight_shape[1]),
                                             dtype = theano.config.floatX),name = layername+'_bias')
        self.bias_vis = theano.shared(np.asarray(self.rng.uniform(-0,0,size = self.Weight_shape[0]),
                                                 dtype = theano.config.floatX),name = layername+'_bias_vis')
        self.layername = layername
        self.core_name = core_name
        self.dropout = dropout
        
        self.params = [self.Weight,self.bias]
        self.pretrain_params = [self.Weight,self.bias,self.bias_vis]

    def output(self,input_x,dropout_flag):
        
        p = T.switch(T.eq(dropout_flag,1),1 - self.dropout,1)
        dropout_noise = self.rng2.binomial(n = 1,p = p, size = (self.Weight_shape[1],))
                                                
        in_x = input_x
        out = self.core(in_x.dot(self.Weight) + self.bias.dimshuffle('x',0))
        out = out * dropout_noise
        
        return out

    def reconstruct(self,input_x):
        
        in_x = input_x
        hid = self.core(in_x.dot(self.Weight) + self.bias.dimshuffle('x',0))
        vis = self.core(hid.dot(self.Weight.T) + self.bias_vis.dimshuffle('x',0))
        
        return hid,vis
        
class RBMLayer(BaseLayer):
    
    def __init__(self,input_shape,Weight_shape,layername = '',core_name = 'sigmoid'):
        
        self.rng = np.random.RandomState(2500)
        self.rng2 = RandomStreams(3000)
        self.input_shape = input_shape
        self.Weight_shape = Weight_shape
        self.Weight = theano.shared(np.asarray(self.rng.uniform(-1,1,size = self.Weight_shape),
                                               dtype = theano.config.floatX),name = layername+'_Weight')
        self.bias = theano.shared(np.asarray(self.rng.uniform(-0,0,size = self.Weight_shape[1]),
                                             dtype = theano.config.floatX),name = layername+'_bias')
        self.bias_vis = theano.shared(np.asarray(self.rng.uniform(-0,0,size = self.Weight_shape[0]),
                                                 dtype = theano.config.floatX),name = layername+'_bias_vis')
        self.layername = layername
        self.core_name = core_name
        
        self.params = [self.Weight,self.bias,self.bias_vis]
        
    
    def vis_to_hid(self,input_x):
        
        out = self.core(input_x.dot(self.Weight) + self.bias.dimshuffle('x',0))
        out = self.rng2.binomial(n = 1,size = out.shape,p = out)
        
        return out
        
    def hid_to_vis(self,hid):
        
        out = self.core(hid.dot(self.Weight.T) + self.bias_vis.dimshuffle('x',0))
        out = self.rng2.binomial(n = 1,size = out.shape,p = out)
        
        return out
        
    def free_energy(self,vis):
        ''' Function to compute the free energy '''
        wx_b = T.dot(vis, self.Weight) + self.bias
        vbias_term = vis.dot(self.bias_vis)
        hidden_term = T.sum(T.log(1 + T.exp(wx_b)), axis=1)
        return -hidden_term - vbias_term

    def hv(self,input_x):
        
        hid = self.vis_to_hid(input_x)
        vis = self.hid_to_vis(hid)
        
        return vis

#test = RBMLayer((100,5),(5,2),'test','sigmoid')
#x = T.matrix('x')
#y = test.output(x)
#func = theano.function([x],y)
#
#xt = np.random.binomial(1,0.5,(100,5))
#yt = func(xt)

class RBM(ADAMNN):
    
    def set_layers(self):
        
        layers = []
        layers.append(RBMLayer((100,10),(10,5),'layer1','sigmoid'))
        self.layers = layers
        self.hid = theano.shared(np.array([[1]],dtype = np.int64),'hid')
                    
    def output(self,input_x,learning_rate):
        
        hid = self.layers[0].vis_to_hid(input_x)
        vis = self.layers[0].hid_to_vis(hid)
        cost = T.mean(self.layers[0].free_energy(input_x)) - T.mean(self.layers[0].free_energy(vis))
        
        grads = T.grad(cost,self.params,consider_constant=[vis])
    
        updates = self.adam_decsent(grads,learning_rate)
        updates.append((self.hid,hid))
        
        return vis,hid,cost,updates
            
if __name__ == '__main__':
    
    
            
    input_x = T.matrix('input_x')
    learning_rate = T.dscalar('learning_rate')
    flag = T.dscalar('flag')
    rbm = RBM()
    vis,hid,loss,updates = rbm.output(input_x,learning_rate)
    func = theano.function([input_x,learning_rate],(vis,hid,loss),updates = updates)
    
    xt = np.random.rand(100,5)
    
    rbm.clear_pars()
    for i in range(100):
        out,loss = func(xt,0.001)
        print(i,loss)
    
    plt.imshow(out[1].reshape((28,28)))
    temp = rbm.hid.get_value()
    plt.imshow(temp[123].reshape((6,6)))
    
    dnn,func = compile_func(DNN)
    
    data = np.sin(np.linspace(0,2*np.pi,400))
    
    xt = data[:-1].reshape((-1,1))
    yt = data[1:].reshape((-1,1))
    
    import time
    dnn.clear_pars()
    start = time.clock()
    for i in range(1000):
        out,loss = func(xt,yt,0.001,1)
        print(i,loss/1000)
    end = time.clock()
    print(end-start)