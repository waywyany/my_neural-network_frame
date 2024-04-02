from Tensor import Tensor
from Parameter import Parameter
import numpy as np

class Module():
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    def parameters(self):
        attr_dict =self.__dict__
        pars=[]
        for key in attr_dict:
            att=attr_dict[key]
            if type(att)==Parameter:
                pars.append(att)
            if hasattr(att,"parameters"):  #如果这个“成员”具有方法parameters说明是个模型，通过递归得到里面的参数
                pars.extend(att.parameters())

        return pars

class Sequential(Module):
    def __init__(self,*args):
        super().__init__()
        self.layers=[]
        for i,layer in enumerate(args):
            self.__dict__[f"{i}"]=layer
            self.layers.append(layer)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

class MouduleList(Module):
    def __init__(self,args):
        super().__init__()
        self.layers=[]
        for i,layer in enumerate(args):
            self.__dict__[f"{i}"]=layer
            self.layers.append(layer)
    def __len__(self):
        return len(self.layers)
    def __getitem__(self, idx):   #获得某一层的信息
        return self.layers[idx]

