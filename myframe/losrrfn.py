from Tensor import Tensor
import numpy as np
from Parameter import Parameter
from Container import Module
from functional import cross_entropy

class CrossEntropyLoss(Module):
    def __init__(self)->None:
        super.__init__()
    def forward(self,x,d):
        y=cross_entropy(x,d)
        return y
