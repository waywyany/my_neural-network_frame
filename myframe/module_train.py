import functional
import layers
import losrrfn
import optim
from Container import*
from Data import*
from layers import*
from losrrfn import*
from optim import*
from Tensor import*
from functional import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid

plt.rcParams['font.sans-serif'] = ['SimSun']  # 用来正常显示中文标签
plt.rcParams['font.size'] = "16"  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


datas=load_img("train-images-idx3-ubyte.gz")
labels=load_label("train-labels-idx1-ubyte.gz")
#这里得到的是类型是array 不是Tensor，放在dataset和dataloader之后才会变成Tensor
#show_single_image(datas[15],labels[15])

datas=datas.astype(np.float32)/225
print(labels[400:410])
#labels=labels.astype(np.float32)

#x1=datas.reshape([-1,784])
#y1=labels.reshape([-1])

# datasets=CustomDataset(datas[2700:3000],labels[2700:3000])
# dataloader=CustomDataLoader(datasets,batch_size=20,shuffle=True)

x1=Tensor(datas[1000:1300])
y1=Tensor(labels[1000:1300])



#x_test=datas.reshape([-1,784])[40:50]
y_test=labels.reshape([-1])[400:440]
x2=Tensor(datas[400:440])
y2=Tensor(y_test)
print(x2.shape)
print(y_test)

# for x,y in dataloader:
#     print(x.data)

class model(Module):
    def __init__(self)->None:
        super().__init__()
        self.layer1=layers.Conv2d(1, 16, 3, 2)
        self.relu1=layers.Relu()
        self.layer2=layers.Conv2d(16,32,3,2)
        self.relu2=layers.Relu()
        self.ft = Flatten()
        self.layer3=layers.Linaer(6*6*32,10)
    def forward(self,x):
        x=x.reshape([-1,1,28,28])
        x=self.layer1(x)
        x=self.relu1(x)
        x=self.layer2(x)
        x=self.relu2(x)
        x=self.ft(x)
        y=self.layer3(x)

        return y
model1=model()

optims=optim.Adam(model1.parameters(),lr=0.001,weight_decay=0.001)
batch_size=20
for e in range(10):
    print(f"current eopch:{e}")
    cnt=0
    for step in range(len(x1)//batch_size):
        x=x1[step*batch_size:(step+1)*batch_size]
        y=y1[step*batch_size:(step+1)*batch_size]
        y_hat=model1(x)
        loss=cross_entropy(y_hat,y)
        loss.backward()
        optims.step()
        optims.zero_grad()
        if step%10==0:
            y_hat=model1(x2)
            p2=y_hat.data.argmax(axis=1)
            print(f"loss:{loss.data}, \nprediction accuracy:({np.mean(p2==y_test)})")
            # print(y_test)
            #print(loss.mean())
    print("-------------------------------------------")


#以下是模型训练完了的测试。
sample1=Tensor(datas[600])
out1=model1(sample1).data.argmax(axis=1)
original1=datas[600]*225
show_single_image(original1,out1)

sample1=Tensor(datas[645])
out1=model1(sample1).data.argmax(axis=1)
original1=datas[645]*225
show_single_image(original1,out1)

sample1=Tensor(datas[789])
out1=model1(sample1).data.argmax(axis=1)
original1=datas[789]*225
show_single_image(original1,out1)

sample1=Tensor(datas[2333])
out1=model1(sample1).data.argmax(axis=1)
original1=datas[2333]*225
show_single_image(original1,out1)

sample1=Tensor(datas[1234])
out1=model1(sample1).data.argmax(axis=1)
original1=datas[1234]*225
show_single_image(original1,out1)

sample1=Tensor(datas[4534])
out1=model1(sample1).data.argmax(axis=1)
original1=datas[4534]*225
show_single_image(original1,out1)

sample1=Tensor(datas[1888])
out1=model1(sample1).data.argmax(axis=1)
original1=datas[1888]*225
show_single_image(original1,out1)

sample1=Tensor(datas[2888])
out1=model1(sample1).data.argmax(axis=1)
original1=datas[2888]*225
show_single_image(original1,out1)




#loss.backward()
# optims.step()
# for i in model1.parameters():
#     print(i.grad)