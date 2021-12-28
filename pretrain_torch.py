import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.io import read_image
import argparse
import cv2
import numpy as np
import os
import time

from torchvision.transforms import transforms


parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=1024, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=112, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=1000, help='number of image channels')
opt = parser.parse_args()
print(opt)

transform = transforms.Compose([
    transforms.Resize(size=[112,112]),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dir = 'GANSketching-main\data\image'

def save_image(img, path, nrow=10):
    img=img[0,:,:,:]
    img=(img+1.0)/2.0*255
    img=img.transpose((1,2,0))
    cv2.imwrite(path,img)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if (classname.find('Conv') != (- 1)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif (classname.find('BatchNorm') != (- 1)):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0.0)

# -----------------------
#  Dataset & Dataloader
# -----------------------

class myDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.dir = img_dir
        self.img_dir = os.listdir(img_dir)
        self.transform = transform
        labels = img_dir.split('/')
        self.label = labels[-1]

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, index):
        img_path = os.path.join(self.dir, self.img_dir[index])
        img = read_image(img_path).float()
        if self.transform:
            img = self.transform(img)
        return img, self.label

train_dir = 'GANSketching-main\data\image\cat'
dataset = myDataset(img_dir=train_dir,transform=transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

class Generator(nn.Module):
    
    def __init__(self, dim=3):
        super(Generator, self).__init__()
        # self.init_size = (opt.img_size // 4)
        # self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, (128 * (self.init_size ** 2))))
        # self.conv_blocks = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128, eps=0.8), nn.LeakyReLU(negative_slope=0.2), nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64, eps=0.8), nn.LeakyReLU(negative_slope=0.2), nn.Conv2d(64, opt.channels, 3, stride=1, padding=1), nn.Tanh())

        # for m in self.modules():
        #     weights_init_normal(m)

        self.fc = nn.Linear(1024, 7*7*256)
        self.fc_bn = nn.BatchNorm2d(256)
        self.deconv1 = nn.ConvTranspose2d(256, 256, 3, 2, 1, 1)
        self.deconv1_bn = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv2_bn = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 256, 3, 2, 1, 1)
        self.deconv3_bn = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm2d(256)
        self.deconv5 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.deconv5_bn = nn.BatchNorm2d(128)
        self.deconv6 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.deconv6_bn = nn.BatchNorm2d(64)
        self.deconv7 = nn.ConvTranspose2d(64 , dim, 3, 1, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        # out = self.l1(z)
        # out = out.view((out.shape[0], 128, self.init_size, self.init_size))
        # img = self.conv_blocks(out)

        x = self.fc(input).reshape((-1, 256, 7, 7))
        x = self.relu(self.fc_bn(x))
        x = self.relu(self.deconv1_bn(self.deconv1(x)))
        x = self.relu(self.deconv2_bn(self.deconv2(x)))
        x = self.relu(self.deconv3_bn(self.deconv3(x)))
        x = self.relu(self.deconv4_bn(self.deconv4(x)))
        x = self.relu(self.deconv5_bn(self.deconv5(x)))
        x = self.relu(self.deconv6_bn(self.deconv6(x)))
        x = self.tanh(self.deconv7(x))
        return x

class Discriminator(nn.Module):
    
    def __init__(self, dim=3):
        super(Discriminator, self).__init__()

        # def discriminator_block(in_filters, out_filters, bn=True):
        #     block = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.25)]
        #     if bn:
        #         block.append(nn.BatchNorm2d(out_filters, eps=0.8))
        #     return block
        # self.model = nn.Sequential(*discriminator_block(opt.channels, 16, bn=False), *discriminator_block(16, 32), *discriminator_block(32, 64), *discriminator_block(64, 128))
        # ds_size = (opt.img_size // (2 ** 4))
        # self.adv_layer = nn.Linear((128 * (ds_size ** 2)), 1)
        
        # for m in self.modules():
        #     weights_init_normal(m)

        self.conv1 = nn.Conv2d(dim, 64, 5, 2, 2)
        self.conv2 = nn.Conv2d(64, 128, 5, 2, 2)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 5, 2, 2)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 5, 2, 2)
        self.conv4_bn = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512*7*7, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        # out = self.model(img)
        # out = out.view((out.shape[0], (- 1)))
        # validity = self.adv_layer(out)

        x = self.leaky_relu(self.conv1(input))
        x = self.leaky_relu(self.conv2_bn(self.conv2(x)))
        x = self.leaky_relu(self.conv3_bn(self.conv3(x)))
        x = self.leaky_relu(self.conv4_bn(self.conv4(x)))
        x = x.reshape((x.shape[0], 512*7*7))
        x = self.fc(x)
        return x

# Initialize generator and discriminator
adversarial_loss = nn.MSELoss()
generator = Generator().cuda()
discriminatorX = Discriminator().cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_DX = torch.optim.Adam(discriminatorX.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

warmup_times = 1
run_times = 1000
total_time = 0.
cnt = 0

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for (i, (real_imgs, _)) in enumerate(dataloader):
        real_imgs = real_imgs.cuda()
        valid = torch.ones([real_imgs.shape[0], 1]).cuda()
        fake = torch.zeros([real_imgs.shape[0], 1]).cuda()


        # -----------------
        #  Generate
        # -----------------

        z = torch.normal(0, 1, (real_imgs.shape[0], opt.latent_dim)).cuda()
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminatorX(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        real_loss = adversarial_loss(discriminatorX(real_imgs), valid)
        fake_loss = adversarial_loss(discriminatorX(gen_imgs.detach()), fake)
        dx_loss = (0.5 * (real_loss + fake_loss))
        dx_loss.backward()
        optimizer_DX.step()

        if warmup_times==-1:
            print(('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]' % (epoch, opt.n_epochs, i, len(dataloader), dx_loss.numpy()[0], g_loss.numpy()[0])))
            batches_done = ((epoch * len(dataloader)) + i)
            if ((batches_done % opt.sample_interval) == 0):
                save_image(gen_imgs.data[:25], ('images/%d.png' % batches_done), nrow=5)
        else:
            cnt += 1
            print(cnt)
            if cnt == warmup_times:
                sta = time.time()
            if cnt > warmup_times + run_times:
                total_time = time.time() - sta
                print(f"run {run_times} iters cost {total_time} seconds, and avg {total_time / run_times} one iter.")
                exit(0)
