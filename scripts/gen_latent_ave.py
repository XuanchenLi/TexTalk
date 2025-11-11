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
    parser.add_argument('--input_dir', type=str, default='/data/your_usrname/TexTalkData/')
    args = parser.parse_args()

    for id in get_ids(args.input_dir):
        print(id)
        frames = get_frames(os.path.join(id, "Models"))
        latent_sum = None
        for idx, frame in tqdm(enumerate(frames)):
            if idx == 0:
                continue
            if not os.path.exists(os.path.join(frame, "latent_gt.pth")):
                break
            latent = torch.load(os.path.join(frame, "latent_gt.pth"))
            latent_sum = latent_sum + latent if latent_sum is not None else latent

        if latent_sum is not None:
            latent_ave = latent_sum / (len(frames) - 1)
            latent_save_path = os.path.join(id, f'latent_motion_ave.pth')
            torch.save(latent_ave.cpu().numpy(), latent_save_path)

        frames = get_frames(os.path.join(id, "Textures_512"))
        latent_sum = None
        for idx, frame in tqdm(enumerate(frames)):
            if idx == 0:
                continue
            if not os.path.exists(os.path.join(frame, "latent_gt.pth")):
                break
            latent = torch.load(os.path.join(frame, "latent_gt.pth"))
            latent_sum = latent_sum + latent if latent_sum is not None else latent

        if latent_sum is not None:
            latent_ave = latent_sum / (len(frames) - 1)
            latent_save_path = os.path.join(id, f'latent_tex_ave.pth')
            torch.save(latent_ave.cpu().numpy(), latent_save_path)


            

if __name__ == '__main__':
    main()
