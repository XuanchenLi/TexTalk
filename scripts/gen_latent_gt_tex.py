import argparse
import cv2
import glob
import numpy as np
import os
import torch
from tqdm import tqdm
from torchvision.transforms.functional import normalize
from basicsr.utils import imwrite, img2tensor, tensor2img
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils.misc import get_device
from basicsr.utils.registry import ARCH_REGISTRY

def recover_map(img_0, w_map):
    w_map = w_map / 2 + 0.5
    res = torch.log((w_map + 1e-8)/(1-w_map + 1e-8)) * img_0
    #res = -torch.log((1 / (w_map + 1e-8)) - 1) * img_0
    #res = torch.logit(w_map, eps=1e-6) * img_0
    assert not torch.any(torch.isnan(res))
    return res


def get_ids(rt):
    subdirs = os.listdir(rt)
    subdirs = [os.path.join(rt,sd) for sd in subdirs]
    subdirs = sorted([sd for sd in subdirs if os.path.isdir(sd) and os.path.exists(os.path.join(sd, "Models"))])
    return subdirs

def get_frames(rt):
    subdirs = os.listdir(rt)
    subdirs = [os.path.join(rt,sd) for sd in subdirs]
    subdirs = sorted([sd for sd in subdirs if os.path.isdir(sd) and os.path.basename(sd).startswith("0")])
    return subdirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default= 'checkpoints/tex_vae.pth'
    )
    parser.add_argument('--input_dir', type=str, default='/data/your_usrname/TexTalkData/')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model
    model = ARCH_REGISTRY.get('VQAutoEncoder')(img_size=512, nf=64, codebook_size=1024, quantizer='nearest', ch_mult=[1, 2, 2, 4, 4, 8], emb_dim=16).to(device)
    model.load_state_dict(torch.load(args.model_path)['params_ema'], strict=True)
    model.eval()
    model = model.to(device)

    for id in get_ids(args.input_dir):
        print(id)
        if not os.path.exists(os.path.join(id, "Textures_512", "000001", "face.png")):
            continue
        img_0 = cv2.imread(os.path.join(id, "Textures_512", "000001", "face.png"), cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img_0 = img2tensor(img_0).cuda()
        
        for frame in tqdm(get_frames(os.path.join(id, "Textures_512"))):
            img_n = cv2.imread(os.path.join(frame, "face.png"), cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_n = img2tensor(img_n).cuda()
            img_diff = torch.sigmoid(img_n/(img_0+1e-5))
            img_diff = (img_diff - 0.5) * 2
            img_diff = img_diff.unsqueeze(0).cuda()
            
            #img_n = img_n.unsqueeze(0).cuda()
            # inference
            with torch.no_grad():
                output = model.encoder(img_diff)
                quant, codebook_loss, quant_stats = model.quantize(output)
                x = model.generator(quant)
                x = tensor2img(x[0])
                #print(output.shape)
            latent_save_path = os.path.join(frame, f'latent_gt.pth')
            torch.save(output.cpu().numpy(), latent_save_path)
            # #cv2.imwrite(os.path.join(frame, f'wrinkle_v2.png'), tensor2img(img_diff[0]))
            # cv2.imwrite(os.path.join(frame, f'latent_img_v3.png'), x)
            
        


if __name__ == '__main__':
    main()