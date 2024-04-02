# my_neural-network_frame
## 这是仿照torch写的一个简单的深度学习框架。
这个框架包括了：自动梯度、module、优化器、损失函数、快速计算卷积的img2col等。  
我用这个框架写了个简单的CNN跑了手写数字识别的数据，成功了，训练三百个数据，测试集准确率80%    

## 这个框架写的匆忙，没细细打包，分类较为粗糙，简单介绍一下
tensor是tensor类，里面重定义了一堆运算符，利用递归实现了自动梯度  
container是模型module基类  
layers是一些具体的模型类，比如linear，con2d
functional是一些算法和激活函数  
lossfn简单写了个交叉熵
optim是两个优化器SGD和Adam
parameter是tensor的一个简单封装
Data类本来是想写出一个类似dataloader的类的，有一些不会的错误，故最后也没用。里面有一个show_single_image可以显示图片。
module_train是简单的一个例子来检验自己的模型能不能用，就是手写数字识别  

### 注：里面的数据文件路径是写死的，我自己方便用的
