# Style-atk （CVPR 2021）

## Description 
Code for the **CVPR21 oral** paper [*Exploring Adversarial Fake Images on Face Manifold*](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Exploring_Adversarial_Fake_Images_on_Face_Manifold_CVPR_2021_paper.html) 

We bypass the State-Of-The-Art fake image detectors via explorihg on Style-GAN's latent space 
We also implement traditional norm-based adversarial attacks on fake image detectors, see in [pgd_baseline](https://github.com/ldz666666/Style-atk/tree/main/pgd_baseline)  

## Abstract
Images synthesized by powerful generative adversarial network (GAN) based methods have drawn moral and privacy concerns. Although image forensic models have reached great performance in detecting fake images from real ones, these models can be easily fooled with a simple adversarial attack. But, the noise adding adversarial samples are also arousing suspicion. In this paper, instead of adding adversarial noise, we optimally search adversarial points on face manifold to generate anti-forensic fake face images. We iteratively do a gradient-descent with each small step in the latent space of a generative model, e.g. Style-GAN, to find an adversarial latent vector, which is similar to norm-based adversarial attack but in latent space. Then, the generated fake images driven by the adversarial latent vectors with the help of GANs can defeat main-stream forensic models. For examples, they make the accuracy of deepfake detection models based on Xception or EfficientNet drop from over 90% to nearly 0%, meanwhile maintaining high visual quality. In addition, we find manipulating style vector z or noise vectors n at different levels have impacts on attack success rate. The generated adversarial images mainly have facial texture or face attributes changing.

### Pipeline
![Image](https://github.com/ldz666666/Style-atk/blob/main/intro_images/image1.png)

### Image Visual Quality Comparsion(qualitatively)
![Image](https://github.com/ldz666666/Style-atk/blob/main/intro_images/image2.png)

### Image Visual Quality Comparsion(quantitatively)
![Image](https://github.com/ldz666666/Style-atk/blob/main/intro_images/table2.png)

### Attack Success Rate
![Image](https://github.com/ldz666666/Style-atk/blob/main/intro_images/table1.png)

## Usage
First, construct a dataset which consists [FFHQ](https://github.com/NVlabs/ffhq-dataset) images and StyleGAN generated images, then train a forensic model, e.g. xception on this dataset.  
To run the attack code, just `python attack_demo.py`  
Test code for both our attack method and norm-based attack method can be found [here](https://github.com/ldz666666/Style-atk/blob/main/pgd_baseline/test_attack.py)  

We use [rosinality's](https://github.com/rosinality/style-based-gan-pytorch) pytorch implementation of StyleGAN, and we refer to [advertorch](https://advertorch.readthedocs.io/en/latest/user/installation.html#latest-version-v0-1) to implement our attack

Weights of forensic models can be found [here](https://pan.baidu.com/s/1vaUdy6BeQtfzgMluUacTdA), password is 269p. 

## Requirements
pytorch 1.5.1  
advertorch 0.1  
opencv-python 4.1.1.26

## Citation
If you found this repo useful, please cite
```
@inproceedings{li2021exploring,
  title={Exploring Adversarial Fake Images on Face Manifold},
  author={Li, Dongze and Wang, Wei and Fan, Hongxing and Dong, Jing},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5789--5798},
  year={2021}
}
```
