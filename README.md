# CompNet Medical

## Installation
This project uses __Python 3.6__ and is tested on CUDA 10.2.

It is highly recommended to create and use a virtual environment to install dependencies and run any code

Either with `virtualenv`:
```
virtualenv ${HOME}/.virtualenv/CompNet
source ${HOME}/.virtualenv/CompNet/bin/activate
```
Or with Anaconda3 (`conda`):
```
conda create -n CompNet python=3.6
conda activate CompNet
```

Clone the repository and install package dependencies:
```
git clone https://github.com/mattesko/CompNet-Medical
cd CompNet-Medical
pip install -r requirements.txt
pip install -e .
```

## References
* Wang, Xiaosong, et al. ChestX-Ray8: Hospital-Scale Chest X-Ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases. p. 10. \[[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)\]
* Jaeger, Stefan, et al. “Two Public Chest X-Ray Datasets for Computer-Aided Screening of Pulmonary Diseases.” Quantitative Imaging in Medicine and Surgery, vol. 4, no. 6, Dec. 2014, pp. 475–77. PubMed Central, doi:10.3978/j.issn.2223-4292.2014.11.20. \[[paper](https://doi.org/10.3978/j.issn.2223-4292.2014.11.20)\]
* A.E. Kavur, N.S. Gezer, M. Barış, P.-H. Conze, V. Groza, D.D. Pham, et al. "CHAOS Challenge - Combined (CT-MR) Healthy Abdominal Organ Segmentation \[[paper](https://arxiv.org/abs/2001.06535)\] \[[dataset](http://doi.org/10.5281/zenodo.3362844)\]
* A.E. Kavur, N.S. Gezer, M. Barış, Y.Şahin, S. Özkan, B. Baydar, et al.  "Comparison of semi-automatic and deep learning-based automatic methods for liver segmentation in living liver transplant donors", Diagnostic and  Interventional  Radiology,  vol. 26, pp. 11–21, Jan. 2020. \[[paper](https://doi.org/10.5152/dir.2019.19025)\]
* Bilic, Patrick, et al. "The liver tumor segmentation benchmark (lits)." arXiv preprint arXiv:1901.04056 (2019). \[[paper](https://arxiv.org/abs/1901.04056)\]
* Kortylewski, Adam, et al. “Compositional Convolutional Neural Networks: A Deep Architecture With Innate Robustness to Partial Occlusion.” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), IEEE, 2020, pp. 8937–46. DOI.org (Crossref), doi:10.1109/CVPR42600.2020.00896. \[[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Kortylewski_Compositional_Convolutional_Neural_Networks_A_Deep_Architecture_With_Innate_Robustness_CVPR_2020_paper.pdf)\]
