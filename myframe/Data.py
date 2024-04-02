import gzip
import os
import pickle
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from Tensor import Tensor
dataset_dir="D:/anythingintest"
train_num=60000
img_dim=(1,28,28)
img_size=784
def load_label(file_name):
    file_path=dataset_dir+"/"+file_name
    print("正在转换labels" + file_name + "到numpy数组")
    with gzip.open(file_path,"rb") as file:
        labels=np.frombuffer(file.read(),np.uint8,offset=8)
    print("ok")
    return labels

def load_img(file_name):
    file_path=dataset_dir+ "/"+file_name

    print("正在转换datas" + file_name +"到numpy数组")
    with gzip.open(file_path,"rb") as file:
        data=np.frombuffer(file.read(),np.uint8,offset=16)
    data=data.reshape(-1,img_size)
    print("ok")
    return data

def show_single_image(image, label):
    # 将图像数组转换为 28x28 的形状，并显示灰度图
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

class CustomDataset:
    def __init__(self, data, labels):
        self.data = Tensor(data)
        self.labels = Tensor(labels)
        self.num_samples = len(data)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class CustomDataLoader:
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
        self.index_array = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.index_array)

    def __iter__(self):
        for i in range(self.num_batches):
            batch_index = self.index_array[i * self.batch_size : (i + 1) * self.batch_size]
            batch_data = [self.dataset[j] for j in batch_index]
            batch_inputs, batch_labels = zip(*batch_data)
            yield Tensor(np.array(batch_inputs)), Tensor(np.array(batch_labels))

    def __len__(self):
        return self.num_batches

