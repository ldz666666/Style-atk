# Style-atk

## Description 
Code for paper *Exploring Adversarial Fake Images on Face Manifold*  
We bypass the State-Of-The-Art fake image detectors via explorihg on Style-GAN's latent space 
We also implement traditional norm-based adversarial attacks on fake image detectors, see in /pgd_baseline  

## Usage
To run the code, just `python attack_demo.py`
Test code for both our attack method and norm-based attack method can be found under /pgd_baseline/test_attack.py 

## Requirements
pytorch 1.5.1
advertorch 0.1
opencv-python 4.1.1.26
