import argparse
import os
import numpy as np
import math
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
# DCGAN  模型
# 参数设定
parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches ')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval betwen image samples')
opt = parser.parse_args()
print(opt)
# 1、准备mnist数据集
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])  #转换 成 batch ×1×28×28   标准化
train_dataset=datasets.MNIST(root="./data",train=True,download=True,transform=transform)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=opt.batch_size)
test_dataset= datasets.MNIST(root="./data",train=False,download=True,transform=transform)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=opt.batch_size)
#2、建立模型（生成器和判别器）
img_shape = (opt.channels, opt.img_size, opt.img_size)    #通道为1  1 *28*28
cuda = True if torch.cuda.is_available() else False
   #2.1 生成器模型
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]   #对传入数据应用线性转换（输入节点数，输出节点数）
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8)) #批规范化
            layers.append(nn.LeakyReLU(0.2, inplace=True))  #加激活函数
            return layers

        self.model = nn.Sequential(     #构造模型
            *block(opt.latent_dim, 128, normalize=False),  #输入100
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),  #np.prod()计算所有元素乘积 1*28*28=784
            nn.Tanh()
        )

    def forward(self, z):   #输入z  噪声信号  64*100
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)   #输出的是64个28*28
        return img
    #2.2 判别器模型
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),   #降为1 个数  分数
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()
# 3、Loss function 和优化器
adversarial_loss = torch.nn.BCELoss()
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# # Configure data loader
# os.makedirs('../../data/mnist', exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST('../../data/mnist', train=True, download=True,
#                    transform=transforms.Compose([
#                        transforms.ToTensor(),
#                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#                    ])),
#     batch_size=opt.batch_size, shuffle=True)


#  Training
for epoch in range(opt.n_epochs):  #训练10次
    for i, (imgs, _) in enumerate(train_loader):
        # （图片 标签）
        # Adversarial ground truths
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)  #计算损失值用  64*1 的向量全为1
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)   #计算损失值    64*1 全为0
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))
    # Train Generator
        optimizer_G.zero_grad()
        #  创建随机噪声
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)  #y_pre label
        g_loss.backward()
        optimizer_G.step()

    #  Train Discriminator
        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(train_loader),
                                                            d_loss.item(), g_loss.item()))
        batches_done = epoch * len(train_loader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)
