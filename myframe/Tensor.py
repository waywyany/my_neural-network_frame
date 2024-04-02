import numpy as np
def data_trans(data):    #transform data to type of ndarray
    if isinstance(data,np.ndarray):return data
    else: return np.array(data)   #将数据data可能是常量转化为array

def tensor_trans(data):
    if isinstance(data,Tensor): return data
    else:return Tensor(data)

class Tensor():
    def __init__(self,data,training=False,depends_on=[],name="input"):
        self._data= data_trans(data)  #确保data都是array
        self.training=training
        self.shape=self._data.shape
        self.grad=None   #type：Tensor
        self.depends_on=depends_on
        self.name=name
        if self.training:
            self.zero_grad()

    def zero_grad(self):
        self.grad=Tensor(np.zeros_like(self.data,dtype=np.float64),training=False)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self,new_data:np.ndarray):   #带参数的方法可以修改data属性
        self._data=new_data

    def __len__(self):
        return len(self._data)

    def __repr__(self):  #打印输出的信息使其一目了然
        return f"Tensor:({self._data},traning={self.training})"

    def __add__(self,other):
        return _add(self,tensor_trans(other))
    def __radd__(self,other):
        return _add(tensor_trans(other),self)

    def __mul__(self, other):
        return _mul(self,tensor_trans(other))
    def __rmul__(self, other):
        return _mul(tensor_trans(other),self)

    def __matmul__(self, other):
        return _matmul(self,tensor_trans(other))
    def __rmatmul__(self, other):
        return _matmul(tensor_trans(other),self)
    def __sub__(self, other):
        return _sub(self,tensor_trans(other))
    def __rsub__(self, other):
        return _sub(tensor_trans(other),self)
    def __truediv__(self, other):
        return _div(self,tensor_trans(other))
    def __neg__(self):
        return _neg(self)

    def __getitem__(self, idx):
        return _getitem(self,idx)

    def __pow__(self,n):
        return _pow(self,n)


    def backward(self,grad=None):
        if grad is None:
            if self.shape==():
                grad= Tensor(1.0)
            else:
                grad=Tensor(np.ones(self.shape))
        self.grad.data= self.grad.data+grad.data
        for temp in self.depends_on:
            tensor,grad_fn=temp
            backward_grad=grad_fn(grad.data)
            tensor.backward(Tensor(backward_grad))

    #Tensor的一些属性功能使用 例如求和 求均值
    def sum(self):
        return tensor_sum(self)
    def mean(self):
        return tensor_mean(self)
    def mm(self,data):
        pass
    def clamp(self,min=0,max=np.inf):
        return _clip(self,min,max)
    def reshape(self,idx):
        return _reshape(self,idx)
    def sigmoid(self):
        return sigmoid(self)
    def tanh(self):
        return tanh(self)
def _add(t1: Tensor,t2: Tensor) ->Tensor:
    data= t1.data+t2.data
    training=t1.training or t2.training
    depends_on=[]  #获得t1和t2的信息
    if t1.training:   #t1的导数就是t2
        def grad_fn1(grad):
            ndims=grad.ndim - t1.data.ndim    #检查多余维度
            for _ in range(ndims):
                grad=grad.sum(axis=0)   #实现降维
            for i,dim in enumerate(t1.shape):
                if dim==1:
                    grad=grad.sum(axis=i,keepdims=True)  #缩掉为1的维数
            return grad
        depends_on.append((t1,grad_fn1))

    if t2.training:
        def grad_fn2(grad):
            ndims=grad.ndim - t2.data.ndim    #检查多余维度
            for _ in range(ndims):
                grad=grad.sum(axis=0)   #实现降维
            for i,dim in enumerate(t2.shape):
                if dim==1:
                    grad=grad.sum(axis=i,keepdims=True)  #缩掉为1的维数
            return grad
        depends_on.append((t2,grad_fn2))

    return Tensor(data,
                  training,
                  depends_on,"add")


def _mul(t1,t2)->Tensor:
    t1=tensor_trans(t1)
    t2=tensor_trans(t2)
    data=t1.data*t2.data
    training=t1.training or t2.training
    depends_on=[]
    if t1.training:
        def grad_fn1(grad:np.ndarray)->np.ndarray:
            grad=grad*t2.data
            ndims=grad.ndim-t1.data.ndim
            for _ in range(ndims):
                grad=grad.sum(axis=0)
            for i,dim in enumerate(t1.shape):
                if dim==1:
                    grad=grad.sum(axis=i,keepdims=True)
            return grad
    depends_on.append((t1,grad_fn1))

    if t2.training:
        def grad_fn2(grad:np.ndarray)->np.ndarray:
            grad=grad*t1.data
            ndims=grad.ndim-t2.data.ndim
            for _ in range(ndims):
                grad=grad.sum(axis=0)
            for i,dim in enumerate(t2.shape):
                if dim==1:
                    grad=grad.sum(axis=i,keepdims=True)
            return grad
    depends_on.append((t2,grad_fn2))

    return Tensor(data,
                  training,
                  depends_on,"mul")

def _div(t1, t2):
    t1 = tensor_trans(t1)
    t2 = tensor_trans(t2)
    data = t1.data / t2.data
    training = t1.training or t2.training
    depends_on = []
    if t1.training:
        def grad_fn1(grad):
            return grad / t2.data
        depends_on.append((t1, grad_fn1))

    if t2.training:
        def grad_fn2(grad):
            return -grad * t1.data / (t2.data ** 2)
        depends_on.append((t2, grad_fn2))

    return Tensor(data, training, depends_on, "div")

def _neg(t:Tensor)->Tensor:
    data=t.data
    dapend_on=[]
    training=t.training
    if t.training:
        depend_on=[(t,lambda x:-x)]
    else:
        depend_on=[]

    return Tensor(data,training,depend_on,"neg")

def _pow(t:Tensor,n)->Tensor:
    data=t.data
    depend_on=[]
    if t.training:
        def grad_fn(grad)->np.ndarray:
            return grad*n*t.data**(n-1)
        depend_on.append((t,grad_fn))
    return Tensor(data,t.training,depend_on,"pow")

def _sub(t1:Tensor,t2:Tensor)->Tensor:
    return t1 + -t2

def _matmul(t1:Tensor,t2:Tensor)->Tensor:
    data=t1.data @ t2.data  #数据做矩阵乘法
    training=t1.training or t2.training
    depends_on=[]
    if t1.training:
        def grad_fn1(grad):
            return grad @t2.data.T
        depends_on.append((t1,grad_fn1))

    if t2.training:
        def grad_fn2(grad):
            return t1.data.T @ grad
        depends_on.append((t2,grad_fn2))

    return Tensor(data,training,depends_on,"matual")

def _getitem(t:Tensor,idxs)->Tensor:
    data=t.data[idxs]
    training=t.training
    if training:
        def grad_fn(grad:np.ndarray)->np.ndarray:
            bigger_grad=np.zeros_like(t.data)
            bigger_grad[idxs]=grad
            return bigger_grad
        depends_on = [(t,grad_fn)]
    else:
        depends_on=[]
    return Tensor(data,training,depends_on,"slice")

def tensor_sum(t:Tensor)->Tensor:
    data=t.data.sum()
    traning=t.training

    if traning:
        def grad_fn(grad):
            return grad*np.ones_like(t.data)
        depends_on=[(t,grad_fn)]
    else:
        depends_on=[]
    return Tensor(data,traning,depends_on,"sum")

def tensor_mean(t:Tensor)->Tensor:
    data=t.data.mean()
    training=t.training
    if training:
        def grad_fn(grad):
            return grad*np.ones_like(t.data)/len(t.data)
        depends_on=[(t,grad_fn)]
    else:
        depends_on=[]

    return Tensor(data,training,depends_on,"mean")

def _reshape(t:Tensor,idxs)-> Tensor:
    ishape=t.data.shape
    data=t.data.reshape(idxs)
    traning=t.training

    if traning:
        def grad_fn(grad):
            bigger_grad=grad.reshape(ishape)  #返回的梯度是原来的形状
            return bigger_grad
        depends_on=[(t,grad_fn)]
    else:
        depends_on=[]

    return Tensor(data,traning,depends_on,"reshape")

def _clip(t:Tensor,smin,smax) ->Tensor:   #限定tensor的数值的范围
    depends_on=[]
    data=t.data
    if t.training:
        def grad_fn(grad:np.ndarray):
            retgrad=np.copy(grad)
            retgrad=retgrad.reshape(-1)
            retgrad[grad.reshape(-1)<smin]=0
            retgrad[grad.reshape(-1)>smax]=0
            return np.reshape(retgrad,grad.shape)
        depends_on.append((t,grad_fn))
    data = np.clip(t.data,smin,smax)
    return Tensor(data,t.training,depends_on,"clip")

def tanh(t1:Tensor) ->Tensor:
    training=t1.training
    depends_on=[]
    data=np.tanh(t1.data)
    if t1.training:
        def grad_fn(grad : np.ndarray)->np.ndarray:
            grad2=np.copy(grad)  #这是什么意思？
            grad2=grad*(1-np.tanh(t1.data)**2)
            return grad2
        depends_on.append((t1,grad_fn))
    return Tensor(data,training,depends_on,"tanh")

def sigmoid(t1:Tensor)->Tensor:
    training=t1.training
    depends_on=[]
    data=1/(1+np.exp(t1.data))
    if t1.training:
        def grad_fn(grad:np.ndarray)->np.ndarray:
            grad2=np.copy(grad)
            grad2=grad*(data*(1-data))
            return grad2
        depends_on.append((t1,grad_fn))
    return Tensor(data,training,depends_on,"sigmoid")

#以下是利用自动梯度对二次函数求极值的演示
# iteration=500
# x=np.random.randn(1)
# lr=0.1
# x1=Tensor(x,training=True)
# for i in range(iteration):
#     output=(x1+32.66)**2
#     output.backward()
#     x1.data-=lr*x1.grad.data  #类似于optim.step()
#     x1.zero_grad()
# print(x1.data)
# print((x1.data+32.66)**2)

