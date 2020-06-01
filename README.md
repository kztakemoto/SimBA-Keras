# SimBA-Keras

This repository contains demo code for [SimBA (Simple Black-box Adversarial Attacks)](https://arxiv.org/abs/1905.07121) implemented in Keras.

## Usage
Install SimBA.
```
pip install git+https://github.com/kztakemoto/adversarial-robustness-toolbox
```
Code for SimBA is [here](https://github.com/kztakemoto/adversarial-robustness-toolbox/blob/master/art/attacks/evasion/simba.py). [nquntan](https://github.com/nquntan) and kaztakemoto implemented SimBA using [Adversarial Robustness 360 Toolbox](https://arxiv.org/abs/1807.01069) (ART) v1.1.

### Non-targeted Attacks
```
python run_demo_simba_single.py
```
![Non-targeted Attacks DCT](assets/plot_nontargeted_dct.png )

### Targeted Attacks
Demo code considers targeted attacks to "ship".
```
python run_demo_simba_single.py --targeted
```
![Targeted Attacks DCT](assets/plot_targeted_dct.png )