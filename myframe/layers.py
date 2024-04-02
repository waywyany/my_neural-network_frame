from Tensor import Tensor
import numpy as np
from Parameter import Parameter
from Container import Module,MouduleList,Sequential
from functional import*
from optim import*
from Tensor import*
class Linaer(Module):
    def __init__(self,nin,nout):
        #super.__init__()
        self.weight= Parameter(np.random.normal(0,1/np.sqrt(nin+nout),[nin,nout]),training=True)
        self.bias = Parameter(np.zeros([nout]),training=True)
    def  forward(self,x):
        y=x@self.weight + self.bias
        return y

class Conv2d(Module):
    def __init__(self,nin,nout,kernel_size=3,stride=1,padding="valid"):
        self.weight=Parameter(
            np.random.normal(0,0.1,[nout,nin,kernel_size,kernel_size]),training=True)
        self.bias=Parameter(np.zeros([1,nout,1,1]),training=True)
        self.stride=stride
        self.pad=padding
    def forward(self,x):
        y=conv2d(x,self.weight,self.stride,self.pad)
        y=y+self.bias
        return y

class Flatten(Module):
    def forward(self,x):
        batch_size = x.shape[0]
        return x.reshape((batch_size,-1))

class Relu(Module):
    def __init__(self)->None:
        super().__init__()
    def forward(self,x):
        y=relu(x)
        return y

class Tanh(Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        y=tanh(x)
        return y

class mynet(Module):
    def __init__(self):
        super(mynet,self).__init__()

        self.ft=Flatten()
        self.fc=Linaer(1323,10)

    def forward(self,x):
        #x=self.layer1(x)
        x=self.ft(x)
        x=self.fc(x)
        return x

# moduel=mynet()
# testdata=np.random.random((100,3,21,21))
# TensorTest=tensor_trans(testdata)
# y=moduel(TensorTest)
# print(y.shape)
# y.backward()
# a=np.ones((6,3,21,21))
# b=a*2
# c=2*np.ones((6,1323))
# at=Tensor(a,training=True)
# bt=Tensor(b,training=True)
# ct=Tensor(c,training=True)
# s=at*bt
# batch_size = s.shape[0]
# s=s.reshape((batch_size,-1))
# y=s*ct
# print(y.shape)
# y.backward()
# print(at.grad)
# print(bt.grad)



