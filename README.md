# CompNet Medical

## Installation
This project uses __Python 3.6__ and is tested on CUDA 10.2.

It is highly recommended to create and use a virtual environment to install dependencies and run any code:
```
virtualenv ${HOME}/.virtualenv/CompNet
source ${HOME}/.virtualenv/CompNet/bin/activate
```

Clone the repositort and install package dependencies:
```
git clone https://github.com/mattesko/CompNet-Medical
cd CompNet-Medical
pip install -r requirements.txt
pip install -e .
```

## References
### Chest X-Ray Data
```
@InProceedings{wang2017chestxray,
    author      = {Wang, Xiaosong and Peng, Yifan and Lu, Le and Lu, Zhiyong and Bagheri, Mohammadhadi and Summers, Ronald},
    title       = {ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases},
    booktitle   = {2017 IEEE Conference on Computer Vision and Pattern Recognition(CVPR)},
    pages       = {3462--3471},
    year        = {2017}
}
```
