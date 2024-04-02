from Tensor import Tensor
import numpy as np

def conv2dbase(x, w, stride):
    """卷积的正向计算"""
    b, c1, h1, w1 = x.shape
    c2, c1, k1, k2 = w.shape
    h2, w2 = (h1-k1)//stride+1, (w1-k2)//stride+1
    # 卷积核心对应位置索引
    idxw = np.arange(k2)[np.newaxis, :] + np.zeros([c1, k1, 1])
    idxh = np.arange(k1)[:, np.newaxis] + np.zeros([c1, 1, k2])
    idxc = np.arange(c1)[:, np.newaxis, np.newaxis] + np.zeros([1, k1, k2])
    idxw = idxw.reshape([1, -1]) + \
        (np.arange(w2) * stride + np.zeros([h2, 1])).reshape([-1, 1])
    idxh = idxh.reshape([1, -1]) + \
        (np.arange(h2)[:, np.newaxis] * stride+np.zeros([w2])).reshape([-1, 1])
    idxc = idxc.reshape([1, -1]) + np.zeros([h2, w2]).reshape([-1, 1])
    idxw = idxw.astype(np.int32)
    idxh = idxh.astype(np.int32)
    idxc = idxc.astype(np.int32)
    w = w.reshape([c2, c1*k1*k2]).T
    #print(w.shape)
    # 此时w的shape为（c1*k1*k2，c2）
    #print(idxc.shape,idxh.shape,idxc.shape)
    # 三个索引的size为（h2*w2，c1*k1*k2）  h2, w2 = (h1-k1)//stride+1, (w1-k2)//stride+1
    col = x[:, idxc, idxh, idxw]
    #print(col.shape)
    #col的size为（N，h2*w2，c1*k1*k2）
    cv = col @ w # 矩阵求导章节，回顾矩阵求导章节
    reim = cv.reshape([b, h2, w2, c2])
    reim = reim.transpose([0, 3, 1, 2])
    return reim, col.reshape([-1, c1*k1*k2])

def conv2d(inputs:Tensor,weight:Tensor,stride=1,padding='same')->Tensor:
    #padding: "same"  or "valid"
    x=inputs.data
    w=weight.data
    b,c1,h1,w1=inputs.shape
    c2, c1, k1, k2 = w.shape
    if padding=='same':
        H2=int((h1-0.1)//stride +1)
        W2=int((w1-0.1)//stride+1)
        #添加0.1的偏移量取整不出错
        pad_h2=k1+(H2-1)*stride -h1
        pad_w2=k2+(W2-1)*stride -w1
        pad_h_left=int(pad_h2//2)
        pad_h_right=int(pad_h2-pad_h_left)
        pad_w_left=int(pad_w2//2)
        pad_w_right=int(pad_w2-pad_h_right)
    elif padding=="valid":
        pad_h_left = 0
        pad_h_right = 0
        pad_w_left = 0
        pad_w_right = 0
    else :
        raise "parameter error"
    xp=np.pad(x, [(0, 0), (0, 0), (pad_h_left, pad_h_right), (pad_w_left, pad_w_right)],'constant',constant_values=0)
    y,col=conv2dbase(xp,w,stride)

    training = inputs.training or weight.training
    depends_on=[]
    if inputs.training:
        def grad_fn(e):
            c2,c1,h2,w2=e.shape
            e = np.repeat(e, stride, axis=2)
            e = np.repeat(e, stride, axis=3)
            hidx = np.arange(h2) * stride
            widx = np.arange(w2) * stride
            # 多出来的部分补0
            for s in range(stride - 1):
                e[:, :, hidx + s + 1, :] = 0
                e[:, :, :, widx + s + 1] = 0
                # 对周围需要补0
            ep = np.pad(e, [(0, 0), (0, 0), (k1 - 1, k1 - 1), (k2 - 1, k2 - 1)],'constant',constant_values = (0,0))
            # 卷积核心翻转180度
            rw = w[:, :, ::-1, ::-1]
            rw = rw.transpose([1, 0, 2, 3])
            # e关于输入x的导数
            dx, colt = conv2dbase(ep, rw, 1)
            # 去除补0的位置
            dx = dx[:, :, pad_h_left:h1 + pad_h_right, pad_w_left:w1 + pad_w_right]
            return dx

        depends_on.append((inputs, grad_fn))
    if weight.training:
        def grad_fn(e):
            ecol=e.transpose([0,2,3,1]).reshape([-1,c2])
            dw=ecol.T @ col
            dw=dw.reshape([c2,c1,k1,k2])
            return dw
        depends_on.append((weight,grad_fn))
    return Tensor(y,training,depends_on)

def relu(t1:Tensor)->Tensor:
    training=t1.training
    depends_on=[]
    if t1.training:
        def grad_fn(grad:np.ndarray)->np.ndarray:
            grad2=np.copy(grad)
            grad2=grad*(t1.data>0).astype(np.float64)
            return grad2
        depends_on.append((t1,grad_fn))
    return Tensor(np.clip(t1.data,0,np.inf),training,depends_on,"relu")

def tanh(t1:Tensor)->Tensor:
    training=t1.training
    depends_on=[]
    data=np.tanh(t1.data)
    if t1.training:
        def grad_fn(grad:np.ndarray)->np.ndarray:
            grad2 = np.copy(grad)
            grad2 =grad*(1-np.tanh(t1.data)**2)
            return grad2
        depends_on.append((t1,grad_fn))
    return Tensor(data,training,depends_on,"tanh")

def sigmoid(t1:Tensor)->Tensor:
    data=1/(1+np.exp(-1.0*t1.data))
    depends_on=[]
    training=t1.training
    if t1.training:
        def grad_fn(grad: np.ndarray) -> np.ndarray:
            grad2 = np.copy(grad)
            grad2 = grad * (data*(1-data))
            return grad2
        depends_on.append((t1, grad_fn))
    return Tensor(data, training, depends_on, "sigmoid")

#损失函数
def cross_entropy(y,d):
    N = len(y)
    e = np.exp(np.clip(y.data, -100, 100))
    e = e/np.sum(e, axis=1, keepdims=True)

    loss = np.sum(-np.log(e[np.arange(N), d.data.astype(int)]))/N
    training = y.training
    if training:
        def grad_fn(grad):
            """参考交叉熵"""
            grad = e.copy()
            grad[np.arange(N), d.data.astype(int)] -= 1
            return grad / N
        depends_on = [(y, grad_fn)]
    else:
        depends_on = []
    return Tensor(loss, training, depends_on)



