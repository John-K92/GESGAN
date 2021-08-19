import os
import torch

from models import Generator
from options import TestOptions
from utils import load_image, to_data, to_var, save_image, gaussian



# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def transform_img(model, imgs, style, opts):
    """
    imgs: 입력받은 test 이미지/이미지들
    style: 각 스타일 idx의 텐서
    """
    for idx, img in enumerate(imgs):
        if opts.gpu:
            img = to_var(img)
        img[:,0:1] = gaussian(img[:,0:1], stddev=0.2)
        result = [model(img, style, test=True)]
        if opts.gpu:
            for i in range(len(result)):              
                result[i] = to_data(result[i])
        for i in range(len(result)):     
            if opts.text_type == 2:  # for gif images
                result_filename = os.path.join(opts.result_dir, str(style.item()), (opts.name+'_'+str(idx)+'.png'))
            else:
                result_filename = os.path.join(opts.result_dir, (str(style.item())+'_'+opts.name+'.png'))
            save_image(result[i][0], result_filename)
    return

def main():
    # parse options
    parser = TestOptions()
    opts = parser.parse()

    print('--- load model ---')
    # If parameter differs from the training phase, the arguments needs to be changed
    model = Generator(n_layers=6, latent_dim=opts.latent_dim, num_domains=opts.num_domains)
    model.load_state_dict(torch.load(opts.model_dir))
    if opts.gpu:
        model.cuda()
    model.eval()
    
    if not os.path.exists(opts.result_dir):
        os.mkdir(opts.result_dir)

    print('--- loading image ---')
    for num in range(opts.num_domains):  # save image for every styles that learnt
        style = torch.tensor([num]).cuda()
        if opts.text_type != 2:
            imgs = [load_image(opts.img_name, opts.text_type, opts.alpha)]
        else:  # for gif images
            imgs = list()  # gif와 일반이미지에 대해 호환되는 코드 작성을 위해 리스트화
            if not os.path.exists(os.path.join(opts.result_dir, str(num))):
                os.mkdir(os.path.join(opts.result_dir, str(num))) 
            for root, _, f_dir in os.walk(opts.img_name):
                for file in f_dir:
                    imgs.append(load_image(os.path.join(root, file), opts.text_type, opts.alpha))

        print('--- save ---')
        transform_img(model, imgs, style, opts)


if __name__ == '__main__':
    main()
