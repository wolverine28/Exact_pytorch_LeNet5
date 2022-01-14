from numpy.core.fromnumeric import resize
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchsummary import summary
from PIL import Image
import torch
from LeNet5 import LeNet5
import torch.optim as optim
import time

# good
numbers = Image.open('num.png')
num_list = []
for i in range(10):
    t = numbers.crop((0+8*i,0,9+8*i,15)).resize((7,12)).convert("L").point(lambda p: p > 110 and 255)
    num_list.append((np.array(t)/255*2-1).flatten())

USE_CUDA = torch.cuda.is_available()
# USE_CUDA = False
print(USE_CUDA)

device = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('학습을 진행하는 기기:',device)

mnist_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(), 
    transforms.Normalize((0.1307,), (0.3081,))
])

mnist_data = datasets.MNIST('./',download=True,transform=mnist_transform,train=True)
mnist_data_ts = datasets.MNIST('./',download=True,transform=mnist_transform,train=False)


data_loader = torch.utils.data.DataLoader(mnist_data,
                                          batch_size=512,
                                          shuffle=True)
ts_data_loader = torch.utils.data.DataLoader(mnist_data_ts,
                                          batch_size=512,
                                          shuffle=True)


net = LeNet5(num_list)
if USE_CUDA:
    net = net.cuda()

summary(net,(1,32,32))


optimizer = optim.Adam(net.parameters(),lr=1e-3)


def train(epoch):
    # loss_track = []
    for e in range(epoch):
        running_loss = []
        running_acc = []
        t = time.time()
        for data,target in data_loader:
            if USE_CUDA:
                data = data.cuda()
                target = target.cuda()
            N = data.size(0)
            optimizer.zero_grad()
            output = net(data)
            loss = net.Loss(target,output)
            loss.backward()
            optimizer.step()

            predicted = torch.argmin(output.data,1)

            running_loss.append(loss.item())
            running_acc.append((predicted==target).sum().item() * (1/N)) 

        batch_loss = np.mean(running_loss)
        batch_acc = np.mean(running_acc)
        print('epoch:{:04d}  loss:{:.3f}  acc:{:.3f}  {:.2f}sec/epoch'.format(e,batch_loss,batch_acc,time.time() - t))
        # loss_track.append(batch_loss)

def test():
    running_acc = []
    for data,target in ts_data_loader:
        if USE_CUDA:
            data = data.cuda()
            target = target.cuda()

        N = data.size(0)
        output = net(data)
        predicted = torch.argmin(output.data,1)
        running_acc.append((predicted==target).sum().item() * (1/N)) 

    batch_acc = np.mean(running_acc)
    print('Test acc:{:.3f}'.format(batch_acc))


def main():
    train(10)
    test()

if __name__ == '__main__':
    main()