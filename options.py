import argparse

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # input data related
        self.parser.add_argument('--img_name', type=str, default='./data/test/blue.png', help='path of the image or directory(separated gif)')
        self.parser.add_argument('--text_type', type=int, default=1, help='0 for distance-based text image, 1 for normal image, 2 for gif images')
        self.parser.add_argument('--alpha', action='store_true', default=True, help='if image contains alpha channel')
        # ouptput related
        self.parser.add_argument('--name', type=str, default='output', help='file name of the outputs')
        self.parser.add_argument('--result_dir', type=str, default='./output', help='path for saving result images')
        # model related
        self.parser.add_argument('--model_dir', type=str, default='./save/GESGAN.ckpt', help='specified the dir of saved texture transfer models')
        self.parser.add_argument('--gpu', action='store_true', default=True, help='Whether using gpu')
        self.parser.add_argument('--latent_dim', type=int, default=256, help='Style code dimension')
        self.parser.add_argument('--num_domains', type=int, default=4, help='Number of domains')
    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt

    
class TrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        
        ## Structure and Texture 
        # data loader realted 
        self.parser.add_argument('--img_dir', type=str, default="/home/JW/AssetGAN/GESGAN/data/train", help='dataset directory')   
        # ouptput related
        self.parser.add_argument('--save_path', type=str, default='/home/JW/AssetGAN/GESGAN/save/', help='model directory to be saved')   
        self.parser.add_argument('--save_name', type=str, default='GESGAN', help='filename of the trained model')          
        # model related
        self.parser.add_argument('--G_nlayers', type=int, default=6, help='number of layers in texture generator G')
        self.parser.add_argument('--D_nlayers', type=int, default=4, help='number of layers in texture discriminator D')  
        self.parser.add_argument('--G_nf', type=int, default=32, help='number of features in the first layer of G')
        self.parser.add_argument('--D_nf', type=int, default=32, help='number of features in the first layer of D')                      
        # training related
        self.parser.add_argument('--epochs', type=int, default=200, help='epoch number of training G')  # 40
        self.parser.add_argument('--batchsize', type=int, default=16, help='batch size')
        self.parser.add_argument('--subimg_size', type=int, default=256, help='size of sub-images, which are cropped from a single image to form a training set')
        self.parser.add_argument('--training_num', type=int, default=2560, help='how many training images in each epoch for transfer')  # 800
        self.parser.add_argument('--gpu', action='store_true', default=True, help='Whether using gpu')
        # Mapping & Labelling related
        self.parser.add_argument('--latent_dim', type=int, default=256, help='Latent vector dimension')
        self.parser.add_argument('--num_domains', type=int, default=4, help='Number of domains')        


    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('--- load options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        return self.opt    