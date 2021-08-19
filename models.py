import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn import Linear, Conv2d, BatchNorm2d, LeakyReLU, ConvTranspose2d, ReLU, Tanh, InstanceNorm2d, ReplicationPad2d
from torch.autograd import Variable


#######################  Texture Network
# based on Convolution-BatchNorm-ReLU
class myConv(nn.Module):
    def __init__(self, num_filter=128, stride=1, in_channels=128):
        super(myConv, self).__init__()
        
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, 
                           stride=stride, padding=0, in_channels=in_channels)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        self.relu = ReLU()
            
    def forward(self, x):
        return self.relu(self.bn(self.conv(self.pad(x))))

class myBlock(nn.Module):
    def __init__(self, num_filter=128, p=0.0):
        super(myBlock, self).__init__()
        
        self.myconv = myConv(num_filter=num_filter, stride=1, in_channels=128)
        self.pad = ReplicationPad2d(padding=1)
        self.conv = Conv2d(out_channels=num_filter, kernel_size=3, padding=0, in_channels=128)
        self.bn = BatchNorm2d(num_features=num_filter, track_running_stats=True)
        self.relu = ReLU()
        self.dropout = nn.Dropout(p=p)
        
    def forward(self, x):
        return self.dropout(self.relu(x+self.bn(self.conv(self.pad(self.myconv(x))))))

class Generator(nn.Module):
    def __init__(self, ngf = 32, n_layers = 5, latent_dim=32, num_domains=2):
        super(Generator, self).__init__()
        self.n = num_domains

        self.label_emb = nn.Embedding(num_domains, latent_dim)

        modelList = []
        modelList.append(ReplicationPad2d(padding=4))
        modelList.append(Conv2d(out_channels=ngf, kernel_size=9, padding=0, in_channels=3))
        modelList.append(ReLU())
        modelList.append(myConv(ngf*2, 2, ngf))
        modelList.append(myConv(ngf*4, 2, ngf*2))
        
        modelList2 = []

        for n in range(int(n_layers/2)): 
            modelList2.append(myBlock(ngf*4, p=0.0))
        # dropout to make model more robust
        modelList2.append(myBlock(ngf*4, p=0.5))
        for n in range(int(n_layers/2)+1,n_layers):
            modelList2.append(myBlock(ngf*4, p=0.0))  
                
        modelList2.append(ConvTranspose2d(out_channels=ngf*2, kernel_size=4, stride=2, padding=0, in_channels=ngf*4))
        modelList2.append(BatchNorm2d(num_features=ngf*2, track_running_stats=True))
        modelList2.append(ReLU())
        modelList2.append(ConvTranspose2d(out_channels=ngf, kernel_size=4, stride=2, padding=0, in_channels=ngf*2))
        modelList2.append(BatchNorm2d(num_features=ngf, track_running_stats=True))
        modelList2.append(ReLU())
        modelList2.append(ReplicationPad2d(padding=1))
        modelList2.append(Conv2d(out_channels=3, kernel_size=9, padding=0, in_channels=ngf))
        modelList2.append(Tanh())
        self.model1 = nn.Sequential(*modelList)
        self.model2 = nn.Sequential(*modelList2)
        
    def forward(self, x, label, test=False):
        batchsize = x.size(0)
        if test:
            if batchsize != len(label):
                label = label.repeat(batchsize)
        label = self.label_emb(label)
        stacked = []
        for x_, label_ in zip(x, label):
            stacked.append(torch.mul(x_, label_))
        x = torch.stack(stacked)

        x = self.model1(x)
        x = self.model2(x)
        return x
    
    
################ Discriminators
# Glyph and Texture Networks: BN
class Discriminator(nn.Module):
    def __init__(self, in_channels, ndf=32, n_layers=3, multilayer=False, IN=False, num_domains=2):
        super(Discriminator, self).__init__()
        self.n = num_domains
        
        modelList = []    
        outlist1 = []
        outlist2 = []
        linearList = []
        kernel_size = 4
        padding = int(np.ceil((kernel_size - 1)/2))
        modelList.append(Conv2d(out_channels=ndf, kernel_size=kernel_size, stride=2,
                              padding=2, in_channels=in_channels))
        modelList.append(LeakyReLU(0.2))

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 4)
            modelList.append(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=2,
                                  padding=2, in_channels=ndf * nf_mult_prev))
            if IN:
                modelList.append(InstanceNorm2d(num_features=ndf * nf_mult))
            else:
                modelList.append(BatchNorm2d(num_features=ndf * nf_mult, track_running_stats=True))
            modelList.append(LeakyReLU(0.2))

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 4)
        outlist1.append(Conv2d(out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult_prev))
        
        outlist2.append(Conv2d(out_channels=ndf * nf_mult, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult_prev))
        if IN:
            outlist2.append(InstanceNorm2d(num_features=ndf * nf_mult))
        else:
            outlist2.append(BatchNorm2d(num_features=ndf * nf_mult, track_running_stats=True))
        outlist2.append(LeakyReLU(0.2))
        outlist2.append(Conv2d(out_channels=1, kernel_size=kernel_size, stride=1,
                              padding=padding, in_channels=ndf * nf_mult))
        linearList.append(Linear(in_features=19*19, out_features=self.n))


        self.model = nn.Sequential(*modelList)
        self.out1 = nn.Sequential(*outlist1)
        self.out2 = nn.Sequential(*outlist2)
        self.multilayer = multilayer
        self.aux_layer = nn.Sequential(*linearList, nn.Softmax())
        
    def forward(self, x):
        y = self.model(x)
        out2 = self.out2(y)
        out3 = out2.clone()
        label = out2.view(out2.size(0), -1)
        label = self.aux_layer(label)
        if self.multilayer:
            out1 = self.out1(y)
            return torch.cat((out1.view(-1), out3.view(-1)), dim=0), label
        else:
            return out3.view(-1), label

    
######################## GESGAN
class GESGAN(nn.Module):
    def __init__(self, G_nlayers = 6, D_nlayers = 5, G_nf = 32, D_nf = 32, latent_dim = 32, num_domains=4, gpu=True):
        super(GESGAN, self).__init__()
        
        self.G_nlayers = G_nlayers
        self.D_nlayers = D_nlayers
        self.G_nf = G_nf
        self.D_nf = D_nf
        self.latent_dim = latent_dim
        self.num_domains = num_domains          
        self.gpu = gpu
        self.lambda_l1 = 100
        self.lambda_gp = 10
        self.lambda_sadv = 0.1
        self.lambda_gly = 1.0
        self.lambda_tadv = 1.0
        self.labmda_cls = 0.1
        self.loss = nn.L1Loss()
        self.n_critic = 2
        self.n_count = 0
        self.init = True

        self.G = Generator(self.G_nf, self.G_nlayers, self.latent_dim, self.num_domains)
        self.D = Discriminator(6, self.D_nf, self.D_nlayers, num_domains=self.num_domains)
        
        self.trainerG = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.trainerD = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))    
    
    # FOR TESTING
    # def forward(self, x, l):
    #     xl[:,0:1] = gaussian(xl[:,0:1], stddev=0.2)
    #     return self.G(xl)
            
    # FOR TRAINING
    # init weight
    def init_networks(self, weights_init):
        self.G.apply(weights_init)
        self.D.apply(weights_init)
        
    # WGAN-GP: calculate gradient penalty 
    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(real_data.shape[0], 1, 1, 1)
        alpha = alpha.cuda() if self.gpu else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)[0]

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() 
                              if self.gpu else torch.ones(disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
    
    def classification_loss(self, logit, target):
        return F.binary_cross_entropy_with_logits(logit, target)
    
    def update_discriminator(self, x, y, label_real):
        with torch.no_grad():
            fake_y = self.G(x, label_real)
            fake_concat = torch.cat((x, fake_y), dim=1)
        fake_output, cls = self.D(fake_concat)
        real_concat = torch.cat((x, y), dim=1)
        real_output, _ = self.D(real_concat)
        gp = self.calc_gradient_penalty(self.D, real_concat.data, fake_concat.data)
        # one-hot encoding으로 변환
        one_hot_label = torch.zeros(len(label_real), self.num_domains)
        one_hot_label[range(len(label_real)), label_real] = 1
        cls_loss = self.classification_loss(cls, one_hot_label.cuda())
        
        LTadv = self.lambda_tadv*(fake_output.mean() - real_output.mean() + self.lambda_gp * gp + cls_loss*self.lambda_cls)
        self.trainerD.zero_grad()
        LTadv.backward()
        self.trainerD.step()
        return (real_output.mean() - fake_output.mean()).data.mean()*self.lambda_tadv + cls_loss.item()

    def update_generator(self, x, y, z, label_real, label_ref, ref=False):
        """
        학습시 1회는 원본, 1회는 다른 스타일의 이미지로 변환;
        그로인한 비교하는 label 이 변환되는 형태
        """
        if ref:
            y = z
            label = label_ref
        else:
            ref = False
            label = label_real
        
        fake_y = self.G(x, label)
        fake_concat = torch.cat((x, fake_y), dim=1)
        fake_output, cls = self.D(fake_concat)
        LTadv = -fake_output.mean()*self.lambda_tadv
        Lrec = self.loss(fake_y, y) * self.lambda_l1
        if ref:
            # 다른 스타일로의 변환 시, 해당 되는 라벨로 one-hot encoding
            # 실제 라벨에 대해서는 discriminator가 학습
            one_hot_label = torch.zeros(len(label), self.num_domains)
            one_hot_label[range(len(label)), label] = 1
            cls_loss = self.classification_loss(cls, one_hot_label.cuda())

        LT = LTadv + Lrec + cls_loss*self.lambda_cls
        self.trainerG.zero_grad()
        LT.backward()
        self.trainerG.step()
        return LTadv.data.mean(), Lrec.data.mean(), cls_loss
    
    def update(self, x, y, z, label_real, label_ref):
        """"
        x: structure 이미지
        y: 원본 이미지 / 스타일 이미지
        z: 스타일 변환된 이미지
        label_real: 원본 이미지의 라벨
        label_ref: 스타일 이미지의 라벨
        휴리스틱하게 D와 G가 매번 같이 update가 되면, 성능이 소폭 하락하는 것을 보임
        때문에, n_critic을 2회로 두어 매 2번 학습마다 generator가 타겟 그리고 원본 라벨로
        복원하는 것을 목표로 함        
        """
        LDadv = self.update_discriminator(x, y, label_real)
        if (self.n_count + 1) % self.n_critic == 0:
            # original-to-target label
            LGadv, Lrec, Lcls = self.update_generator(x, y, z, label_real, label_ref, ref=True)

            # original-to-original label
            LGadv_, Lrec_, Lcls_ = self.update_generator(x, y, z, label_real, label_ref)
            LGadv += LGadv_
            Lrec += Lrec_
            Lcls += Lcls_
        else:
            LGadv = 0
            Lrec = 0
            Lcls = 0
        self.n_count+=1
        return [LDadv, LGadv, Lrec, Lcls]
    
    def save_model(self, filepath, filename, epoch=0):
        if epoch != 0:
            torch.save(self.G.state_dict(), os.path.join(filepath, filename+'-G.ckpt'))
            torch.save(self.D.state_dict(), os.path.join(filepath, filename+'-D.ckpt'))
        else:
            torch.save(self.G.state_dict(), os.path.join(filepath, filename+'-G.ckpt'))
            torch.save(self.D.state_dict(), os.path.join(filepath, filename+'-D.ckpt'))

