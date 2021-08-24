import os
import re
import cv2
import random
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as pyimg
from PIL import Image

import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset

pil2tensor = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5])])
tensor2pil = transforms.ToPILImage()

def to_var(x):
    """Convert tensor to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def to_data(x):
    """Convert variable to tensor."""
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data

def visualize(img_arr):
    plt.imshow(((img_arr.numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8))
    plt.axis('off')
    
def load_image(filename, load_type=0, alpha=False):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5,0.5,0.5]),
    ])
    
    if load_type == 0:  # 학습의 structure 이미지와 같은 distance-based의 경우
        img = Image.open(filename).convert('RGB')
        # size = img.size
        img = img.resize((256, 256))
        img = transform(img)
        save_image(img, '../tmp.png')
        return img.unsqueeze(dim=0)
    else:  # 일반적인 이미지 또는 alpha 채널이 있는 이미지의 경우
        img = text_image_preprocessing(filename, alpha)
    img = transform(img)
    return img.unsqueeze(dim=0)

def remove_transparency(im, bg_colour=(0, 0, 0)):
        # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        # Need to convert to RGBA if LA format due to a bug in PIL (http://stackoverflow.com/a/1963146)
        alpha = im.convert('RGBA').getchannel('A')
        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format
        # (http://stackoverflow.com/a/8720632  and  http://stackoverflow.com/a/9459208)
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        bg = bg.convert('RGB')
        return bg

    else:
        return im

def save_image(img, filename):
    tmp = ((img.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)
    cv2.imwrite(filename, cv2.cvtColor(tmp, cv2.COLOR_RGB2BGR))


def text_image_preprocessing(filename, alpha):
    """
    이미지를 distance-based text image로 변경;
    흑백 이미지일때 최대 극대화를 기준으로 사용
    """
    if not alpha:
        I = np.array(Image.open(filename).convert('RGB'))
    else:
        I = Image.open(filename)
        I = np.array(remove_transparency(I))
    I = cv2.resize(I, (256, 256), interpolation=cv2.INTER_CUBIC)
    BW = I[:,:,0] > 127
    G_channel = pyimg.distance_transform_edt(BW)
    G_channel[G_channel>32]=32
    B_channel = pyimg.distance_transform_edt(1-BW)
    B_channel[B_channel>200]=200
    I[:,:,1] = G_channel.astype('uint8')
    I[:,:,2] = B_channel.astype('uint8')
    return Image.fromarray(I)

def gaussian(ins, mean = 0, stddev = 0.2):
    noise = ins.data.new(ins.size()).normal_(mean, stddev)
    return torch.clamp(ins + noise, -1, 1)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('my') == -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def pixlate(image, pixel_size=4):
    image = image.resize(
        (image.size[0] // pixel_size, image.size[1] // pixel_size),
        Image.NEAREST
    )
    image = image.resize(
        (image.size[0] * pixel_size, image.size[1] * pixel_size),
        Image.NEAREST
    )
    return image

# prepare batched filenames of all training data
# [[list of file names in one batch],[list of file names in one batch],...,[]]


    

# for texture transfer:  [Input,Output]=[X, Y]
class ImageStyleFolder(Dataset):
    def __init__(self, opts, no_steps, transforms=pil2tensor):
        """
        X: structure 이미지; X={cls_idx:이미지}
        Y: 원본 이미지 / 스타일 이미지; Y={cls_idx:이미지}
        Z: 스타일 변환된 이미지; Z={cls_idx:{ref_idx:이미지}}
        """
        self.root_dir = opts.img_dir
        self.classes = os.listdir(self.root_dir)
        self.transforms = transforms
        self.X = dict()
        self.Y = dict()
        self.Z = dict()
        self.Noise = []
        self.labels = []
        self.size = no_steps*opts.batchsize  # number of iter
        self.opts = opts
        if opts.gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        for idx, cls in enumerate(self.classes):
            self.Z[idx] = dict()
            cls_dir = os.path.join(self.root_dir, cls)
            for img_path in glob(os.path.join(cls_dir, '*.png')):
                if len(re.findall(r"\d+", img_path.split('/')[-1])):
                    # 원본 이미지에서 스타일 변환된 이미지 추출과정
                    name = img_path.split('/')[-1].split('2')[-1][:-4]
                    ref_idx = self.classes.index(name)
                    img = Image.open(img_path)
                    img = img.resize((400, 400))
                    Z = pil2tensor(img).unsqueeze(dim=0).to(self.device)
                    self.Z[idx][ref_idx] = (Z)
                else:
                    # 원본 이미지의 경우, 좌측은 distance-based 그리고 원본 이미지(y)로 구성되어
                    # 이를 나누는 과정
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((800, 400))
                    self.ori_wd, self.ori_ht = img.size
                    ori_wd = self.ori_wd // 2
                    X = pil2tensor(img.crop((0,0,ori_wd,self.ori_ht))).unsqueeze(dim=0).to(self.device)
                    Y = pil2tensor(img.crop((ori_wd,0,ori_wd*2,self.ori_ht))).unsqueeze(dim=0).to(self.device)
                    self.X[idx] = X
                    self.Y[idx] = Y

                    # _X = pil2tensor(pixlate(img).crop((0,0,ori_wd,self.ori_ht))).unsqueeze(dim=0).to(self.device)
                    # self.X[idx].append(_X)
                    # _Y = pil2tensor(pixlate(img).crop((ori_wd,0,ori_wd*2,self.ori_ht))).unsqueeze(dim=0).to(self.device)
                    # self.Y[idx].append(_Y)
            self.labels.append(idx)



    def __getitem__(self, idx): 
        """
        x: original shape image
        y: original image
        z: ref_idx의 이미지
        cls_idx: 해당 이미지 라벨(아이디)
        ref_idx: 변환시키고자하는 클래스의 아이디
        
        데이터로더는 데이터셋에서 한셋의 이미지로 추출하여 데이터로더를 생성
        x, y, z, cls_idx, ref_idx로 변환 과정
        """
        cls_idx = random.randint(0,len(self.classes)-1)
        ref_idx = list(range(len(self.classes)))
        ref_idx.remove(cls_idx)
        # 원본을 제외하고 랜덤하게 reference될 라벨 idx 추출
        ref_idx = random.choice(ref_idx)
        # random_pix = random.random()  # 랜덤하게 픽셀아트화
        # if random_pix >= 0.3:  # 30프로는 픽셀을 기반으로 학습
        #     x, y, z = self.X[cls_idx][0], self.Y[cls_idx][0], self.Z[cls_idx][ref_idx][0]
        # else:
        x, y, z = self.X[cls_idx], self.Y[cls_idx], self.Z[cls_idx][ref_idx]
        Noise = torch.tensor(0).float().repeat(1, 1, 1).expand(3, self.ori_ht, self.ori_wd//2)
        Noise = Noise.data.new(Noise.size()).normal_(0, 0.2)
        Noise = Noise.unsqueeze(dim=0)
        x = to_var(x) if self.opts.gpu else x
        y = to_var(y) if self.opts.gpu else y
        z = to_var(z) if self.opts.gpu else z
        Noise = to_var(Noise) if self.opts.gpu else Noise
        x, y, z= _cropping_training_batches(x, y, z, Noise, self.opts.subimg_size, self.opts.subimg_size)
        return x, y, z, cls_idx, ref_idx

    def __len__(self):
        return self.size


def _cropping_training_batches(Input, Output, Ref, Noise, wd=256, ht=256):
    """
    Input: X
    Output: Y
    Ref: Z

    랜덤하게 256의 이미지 크기로 crop하여 이미지를 학습;
    이를 위해 각 X, Y, Z의 동일한 위치를 추출하는 과정
    """
    ori_wd = Input.size(2)
    ori_ht = Input.size(3)
    w = random.randint(0,ori_wd-wd)
    h = random.randint(0,ori_ht-ht)
    input = Input[:,:,w:w+wd,h:h+ht].clone()
    output = Output[:,:,w:w+wd,h:h+ht]
    ref = Ref[:,:,w:w+wd,h:h+ht]
    noise = Noise[:,:,w:w+wd,h:h+ht]
    input[:,0] = torch.clamp(input[:,0] + noise[:,0], -1, 1)        
    return input, output, ref
