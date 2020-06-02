# SimBA-Keras

This repository contains demo code for [SimBA (Simple Black-box Adversarial Attacks)](https://arxiv.org/abs/1905.07121) implemented in Keras.

Demo uses the VGG-20 model for the CIFAR-10 dataset obtained from a GitHub repository [GuanqiaoDing/CNN-CIFAR10](https://github.com/GuanqiaoDing/CNN-CIFAR10)

## Usage
### Install SimBA
```
pip install git+https://github.com/kztakemoto/adversarial-robustness-toolbox
```
Code for SimBA is [here](https://github.com/kztakemoto/adversarial-robustness-toolbox/blob/master/art/attacks/evasion/simba.py). [nquntan](https://github.com/nquntan) and kaztakemoto implemented SimBA using [Adversarial Robustness 360 Toolbox](https://arxiv.org/abs/1807.01069) (ART) v1.1.

### Non-targeted Attacks
Attacks using discrete cosine transform (DCT) basis vectors are used in default.
```
python run_demo_simba_single.py
```
![Non-targeted Attacks DCT](assets/plot_nontargeted_dct.png)

Random pixel attaks are also available.
```
python run_demo_simba_single.py --attack px
```
![Non-targeted Attacks pixel](assets/plot_nontargeted_pixel.png)

### Targeted Attacks
Demo code considers targeted attacks to "ship".
```
python run_demo_simba_single.py --targeted
```
![Targeted Attacks DCT](assets/plot_targeted_dct.png)