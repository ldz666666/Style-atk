# Style-atk

## Description 
Code for paper *Exploring Adversarial Fake Images on Face Manifold*  
We bypass the State-Of-The-Art fake image detectors via explorihg on Style-GAN's latent space 
We also implement traditional norm-based adversarial attacks on fake image detectors, see in [pgd_baseline](https://github.com/ldz666666/Style-atk/tree/main/pgd_baseline)  

## Usage
To run the code, just `python attack_demo.py`  
Test code for both our attack method and norm-based attack method can be found under [here](https://github.com/ldz666666/Style-atk/blob/main/pgd_baseline/test_attack.py)  

We use [rosinality's](https://github.com/rosinality/style-based-gan-pytorch) pytorch implementation of StyleGAN, and we refer to [advertorch](https://advertorch.readthedocs.io/en/latest/user/installation.html#latest-version-v0-1) to implement our attack

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
