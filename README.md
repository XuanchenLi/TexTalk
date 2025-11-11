# TexTalker: Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture (CVPR 2025)

<a href='https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Towards_High-fidelity_3D_Talking_Avatar_with_Personalized_Dynamic_Texture_CVPR_2025_paper.pdf'><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2310.00434-red?link=https%3A%2F%2Farxiv.org%2Fabs%2F2310.00434"></a>
<a href='https://xuanchenli.github.io/TexTalk/'><img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-blue?logo=github&labelColor=black&link=https%3A%2F%2Fraineggplant.github.io%2FDiffPoseTalk"></a>

---
## Abstract
![teaser](./figs/teaser.png)

Significant progress has been made for speech-driven 3D face animation, but most works focus on learning the motion of mesh/geometry, ignoring the impact of dynamic texture. In this work, we reveal that dynamic texture plays a key role in rendering high-fidelity talking avatars, and introduce a high-resolution 4D dataset TexTalk4D, consisting of 100 minutes of audio-synced scan-level meshes with detailed 8K dynamic textures from 100 subjects. Based on the dataset, we explore the inherent correlation between motion and texture, and propose a diffusion-based framework TexTalker to simultaneously generate facial motions and dynamic textures from speech. Furthermore, we propose a novel pivot-based style injection strategy to capture the complicity of different texture and motion styles, which allows disentangled control. TexTalker, as the first method to generate audio-synced facial motion with dynamic texture, not only outperforms the prior arts in synthesising facial motions, but also produces realistic textures that are consistent with the underlying facial movements.

---

## TexTalk4D Dataset
Google Drive: Coming Soon

- TexTalkData.zip: Containing 70 seen IDs. The latter half of the speech from ID063 to ID072 is not included in the training set and is used for calculating quantitative metrics, i.e., TexTalk4D-Test-A in the paper.
- TexTalkTest.zip: Containing 18 unseen IDs used for qualitative evaluation.
- TexTalkDataV2.zip: Containing 8 unused IDs. The original dataset lacks subjects with forehead wrinkles. Therefore, we have added 8 IDs who exhibit this feature to support future research."

  Due to storage and ethical constraints, the download link only provides textures with a resolution of 512. Please contact us with your identification details if you need higher-resolution data.

  Some subjects requested to withdraw their data, resulting in a discrepancy between the currently available dataset and the one described in the publication.
---

## Install

```bash
conda create -n textalker python=3.9 -y
conda activate textalker

pip install -r requirements.txt
# basicsr
python basicsr/setup.py develop
# face3d
git clone https://github.com/YadiraF/face3d
cd face3d/mesh/cython
python setup.py build_ext -i 
```

## Train
### Preprocess your dataset
Note: The organization of the dataset directory should be the same as the data we provide.

1. Generate the offset map of the mesh sequence. 
```bash
python ./scripts/gen_obj_diff.py --input_dir "path to your training set root"
```
2. Train the motion vae.
Modify the options/VQGAN_Motion.yml, set dataroot_gt: "/data/your_usrname/TexTalkData".
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_train.sh 4 options/VQGAN_Motion.yml
```
3. Train the texture vae.
Modify the options/VQGAN_Texture.yml, set dataroot_gt: "/data/your_usrname/TexTalkData".
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_train.sh 4 options/VQGAN_Texture.yml
```
4. Generate motion latent features using the motion vae.
```bash
python ./scripts/gen_latent_gt_mesh.py --input_dir "path to your training set root" \ 
--model_path "path to your motion vae checkpoint.pth"
```
5. Generate texture latent features using the texture vae.
```bash
python ./scripts/gen_latent_gt_tex.py --input_dir "path to your training set root" \ 
--model_path "path to your texture vae checkpoint.pth"
```
6. Generate the pivot features.
```bash
python ./scripts/gen_latent_ave.py --input_dir "path to your training set root"
```
7. Train the textalker model.
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
./scripts/dist_train.sh 4 options/TexTalker.yml 
```

## Inference
### Quick start
1. Download the checkpoints and put them to ./checkpoints.
Google Drive: Coming Soon
2. Run the inference.
Require the template obj, texture, motion pivot and texture pivot.
```bash
python inference.py --tex_decoder_path "checkpoints/tex_vae.pth" \
--motion_decoder_path "checkpoints/motion_vae.pth" \
--model_path "checkpoints/textalker.pth" \
--input "example/test_id" \
--audio_path "example/Records/enhanced_vocal.wav" \
--output "results"
```
You can use your own template mesh and generate the pivots by Steps 4-6 of the training stage.


## Acknowledgement
This work is built on awesome research works and open-source projects, thanks a lot to all the authors.
- [DiffPoseTalk](https://github.com/DiffPoseTalk/DiffPoseTalk)
- [BasicSR](https://github.com/XPixelGroup/BasicSR)
- [Face3D](https://github.com/yfeng95/face3d)

---
## Citation	
If our work is useful for your research, please consider citing:
```
@inproceedings{li2025towards,
  title={Towards High-fidelity 3D Talking Avatar with Personalized Dynamic Texture},
  author={Li, Xuanchen and Wang, Jianyu and Cheng, Yuhao and Zeng, Yikun and Ren, Xingyu and Zhu, Wenhan and Zhao, Weiming and Yan, Yichao},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={204--214},
  year={2025}
}
```
