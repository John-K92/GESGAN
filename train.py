from __future__ import print_function
import torch

from models import GESGAN
from utils import weights_init
from utils import ImageStyleFolder
from options import TrainOptions


def main():
    # parse options
    parser = TrainOptions()
    opts = parser.parse()

    # create model
    print('--- create model ---')
    model = GESGAN(opts.G_nlayers, opts.D_nlayers, \
                    opts.G_nf, opts.D_nf, opts.latent_dim, opts.num_domains)

    if opts.gpu:
        print("gpu on!!")
        model.cuda()
    model.init_networks(weights_init)
    model.train()

    print('--- training ---')
    # 데이터를 매 에폭 별 재생성하며 진행하기에 no_steps를 따로 구함
    no_steps = opts.training_num//opts.batchsize
    dataset = ImageStyleFolder(opts, no_steps)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opts.batchsize,
										 shuffle = True)

    for epoch in range(opts.epochs):
        # 구한 no_steps를 기반으로 에폭 반복문안에서 데이터로더를 활용
        for i, data in enumerate(dataloader):
            x_real, y_ref, z_ref, label_real, label_ref = data
            losses = model.update(x_real.squeeze(1), y_ref.squeeze(1),
                        z_ref.squeeze(1), label_real.cuda(), label_ref.cuda()) 
            print('Step1, Epoch [%02d/%02d][%03d/%03d]' %(epoch+1, opts.epochs, i+1,
                                                         opts.training_num//opts.batchsize), end=': ')                                                         
            print('LDadv: %+.3f, LGadv: %+.3f, Lrec: %+.3f, Lcls: %+.3f'%(losses[0], losses[1], losses[2], losses[3])) 
            break
        if epoch and epoch % 10 == 0:
            model.save_model(opts.save_path, opts.save_name + str(epoch) )
        
    print('--- save ---')
    model.save_model(opts.save_path, opts.save_name)   

if __name__ == '__main__':
    main()
