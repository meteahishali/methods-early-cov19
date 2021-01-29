Methods for Advance Detection of COVID-19
=============================

This repository includes the implentation of the methods in [Advance Warning Methodologies for COVID-19 using Chest X-Ray Images](https://arxiv.org/abs/2006.05332).

Software environment:
```
1. Python with the following version and libraries.
python == 3.7.9
tensorflow==2.1.0
Numpy == 1.19.2
SciPy == 1.5.2
scikit-learn == 0.23.2
imageio == 2.9.0
numba == 0.52.0
argparse == 1.1
tqdm == 4.56.0
```
```
2. MATLAB -> MATLAB R2019a.
```

- [Getting started with Early-QaTa-COV19 Dataset](#Getting-started-with-Early-QaTa-COV19-Dataset)
- [Compact Approaches](#Compact-Approaches)
	- [Feature Extraction by DenseNet-121](#Feature-Extraction-by-DenseNet-121)
	- [Sparse Representation based Classification (SRC)](#Sparse-Representation-based-Classification-SRC)
    - [Collaborative Representation based Classification (CRC)](#Collaborative-Representation-based-Classification-CRC)
    - [Convolutional Support Estimator Networks (CSENs)](#Convolutional-Support-Estimator-Networks-CSENs)
    - [Multi-Layer Perceptrons (MLPs)](#Multi-Layer-Perceptrons-MLPs)
    - [Support Vector Machine (SVM)](#Support-Vector-Machine-SVM)
    - [k-Nearest Neighbor (k-NN)](#k-Nearest-Neighbor-k-NN)
- [Deep Learning based Classification](#Deep-Learning-based-Classification)
- [Disclaimer](#Disclaimer)
- [Citation](#Citation)
- [References](#References)

## Getting started with Early-QaTa-COV19 Dataset

Early-QaTa-COV19 dataset is a subset of QaTa-COV19 dataset consisting of 1065 chest X-rays including no or limited sign of COVID-19 pneumonia cases for early COVID-19 detection.

Download the QaTa-COV19 dataset from the link below or using the Kaggle API: https://www.kaggle.com/aysendegerli/qatacov19-dataset

```
kaggle datasets download aysendegerli/qatacov19-dataset
unzip qatacov19-dataset.zip
```

The provided ```create_numpy_data.py``` script is used to arrange the data in numpy arrays for the methods in this repository:
```
cd methods_early-cov19/
mkdir data/
python create_numpy_data.py \
    --dir_early QaTa-COV19/Early-QaTa-COV19/No_COVID-19_Pneumonia_Cases/ \
    --dir_mild QaTa-COV19/Early-QaTa-COV19/Limited_COVID-19_Pneumonia_Cases/ \
    --dir_control QaTa-COV19/Control_Group/Control_Group_I/
```

## Compact Approaches

```
cd methods_early-cov19/compact_classifiers/
```

### Feature Extraction by DenseNet-121

Methods in this group need an additional step which is the feature extraction. DenseNet-121 [1] model pre-trained on Early-QaTa-COV19 is used for this purpose. We extract 1024-D feature vectors after global pooling operation just before the classification layer of the DenseNet-121. Then, a dimensionality reduction is applied over the computed features with PCA by choosing the first 512 principal components.

The pretrained network on Early-QaTa-COV19 is provided: [DenseNet-121](https://drive.google.com/file/d/1QFiuE_odVLSbLvwNNF8n8xyBL-d1IX0a/view?usp=sharing). After downloading the model, save the model under ```Models/``` folder. Now, features can be extracted:
```
python feature_extraction.py --weights True
```
In fact, we already provide the extracted [features](https://drive.google.com/file/d/10BmfLgP1FZ8_pjfGSh_9rO1Ye444uTfl/view?usp=sharing). After downloading, store them under ```/features``` for the classifiers in this group. Alternatively, you may choose to train DenseNet-121 from scratch (ImageNet weights) on Early-QaTa-COV19 for the feature extraction:
```
python feature_extraction.py
```

### Sparse Representation based Classification (SRC)

```
cd methods_early-cov19/compact_classifiers/src
```
For SRC approach, two methods are included in ```main_Demo.m``` script: Dalm [2] and Homotopy [3]. Since they are computationally expensive, the light version with smaller dictionary size can be used by setting ```dicSizeLight=1```. The main script can be run as follows on MATLAB:
```
run main_Demo.m
```

### Collaborative Representation based Classification (CRC)

```
cd methods_early-cov19/compact_classifiers/crc_csen
```

Similarly, the main script can be run as follows on MATLAB:

```
run main_Demo.m
```

The CRC approach [4] is faster compared to l1-minimization recovery algorithms that work in iterative manner (SRC methods). Hence, you may set ```versionCRC = 'heavy'``` in ```main_Demo.m``` to include all the training samples in the dictionary.

Additionally, once the CRC approach is run, the preparation for the next approach, CSEN, is performed and the input and output pairs of CSENs are stored under ```/CSENdata```. Please not the fact that CSENs require only the light version of CRC with a limited number of samples in the dictionary: ```versionCRC = 'light'``` should be set to activate CSEN preparation.

### Convolutional Support Estimator Networks (CSENs)

```
cd methods_early-cov19/compact_classifiers/crc_csen
```

CSEN type of networks take the computed proxy as input (the least-square sense solution). Hence, please run [CRC approach](#Collaborative-Representation-based-Classification-(CRC)) first for the prepration of CSEN classifiers.

There are implemented three networks in the form of CSEN configuration:
- CSEN1 with two hidden layers having 48 and 24 neurons.
- CSEN2 with additional max-pooling and transposed-convolutional layers having 24 neurons.
- ReconNet: structure as in [5], but modiﬁed to perform support estimation as a deep version of the CSEN framework.

The CSEN approach with the provided weights can be evaluated on Early-QaTa-COV19 as follows:
```
python csen.py --test True
```
Networks can be trained from scratch by not passing ```test``` argument. Similarly, ```reconnet.py``` script can be run to evaluate ReconNet in CSEN configuration.

### Multi-Layer Perceptrons (MLPs)

```
cd methods_early-cov19/compact_classifiers/mlp
```
MLP network is implemented with 3-hidden layers having 512, 256, and 64 neurons, respectively. The first layer is initialized with the PCA matrix computed over the training set. To train the MLP network and evaluate on Early-QaTa-COV19, simple run the following script: 
```
python mlp.py
```

### Support Vector Machine (SVM)

```
cd methods_early-cov19/compact_classifiers/svm
```

Evaluation of SVM classifier on the Early-QaTa-COV19 dataset can be performed by running the provided following script:

```
python main.py
```

### k-Nearest Neighbor (k-NN)

```
cd methods_early-cov19/compact_classifiers/knn
```

Similarly, evaluation of k-NN classifier on Early-QaTa-COV19 can be performed by running the provided following script on MATLAB:

```
run main_Demo.m
```


|Method|Accuracy|Sensitivity|Specificity|
|:---:|:---:|:---:|:---:|
|SRC-Dalm| 0.9852| 0.8864| 0.9935|
|SRC-Hom.| 0.9778| 0.9211| 0.9826|
|CRC-light| 0.9730| 0.9559| 0.9744|
|CSEN1| 0.9513| 0.970| 0.9497|
|CSEN2| 0.9566| 0.9728| 0.9552|
|ReconNet| 0.9322| 0.9662| 0.9293|
|MLP| 0.9699| 0.9352| 0.9728|
|SVM| 0.9830| 0.8892| 0.9910|
|k-NN| 0.9741| 0.9305| 0.9778|

## Deep Learning based Classification

In this group, we investigate three recent deep models: DenseNet-121 [1], ResNet-50 [6], and Inception-v3 [7].
```
cd methods_early-cov19/deep_network_classification/
```
### Training
To train a network, run ```eval.py``` by passing the model name (i.e., DenseNet121, InceptionV3, ResNet50) as follows,
```
python eval.py --model DenseNet121
```
### Pretrained Models
We provide the trained networks on Early-Qata-COV19:
* [DenseNet-121](https://drive.google.com/file/d/1QFiuE_odVLSbLvwNNF8n8xyBL-d1IX0a/view?usp=sharing)
* [Inception-v3](https://drive.google.com/file/d/18k3_EwVOFxLLVIffrW1flBv06pKaywxm/view?usp=sharing)
* [ResNet-50](https://drive.google.com/file/d/1RJjUV2fkCrekwyT1g2h31f4IIVI6xdWC/view?usp=sharing)

After downloading the pretrained models, save them under ```/Models```. Now, they can be evaluated by setting test flag.
```
python eval.py --model DenseNet121 --test True
```

|Method|Accuracy|Sensitivity|Specificity|
|:---:|:---:|:---:|:---:|
|DenseNet-121| 0.9937| 0.9502| 0.9974|
|Inception-v3| 0.9791| 0.8469| 0.9904|
|ResNet-50| 0.9884| 0.9155| 0.9946|

## Disclaimer

All the provided implementation of the methods and their source codes are intended to be used for educational and research purposes only, for example, not for clinical applications or as diagnostic tools.

## Citation
If you use method(s) provided in this repository, please cite the following paper:
```
@misc{ahishali2020advance,
      title={Advance Warning Methodologies for COVID-19 using Chest X-Ray Images}, 
      author={Mete Ahishali and Aysen Degerli and Mehmet Yamac and Serkan Kiranyaz and Muhammad E. H. Chowdhury and Khalid Hameed and Tahir Hamid and Rashid Mazhar and Moncef Gabbouj},
      year={2020},
      eprint={2006.05332},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
}
```

## References
[1] G. Huang, Z. Liu, L. Van Der Maaten, and K. Q. Weinberger, “Densely connected convolutional networks,” *in Proc. IEEE Conf. Comput. Vision and Pattern Recognit. (CVPR)*, 2017, pp. 4700–4708. \
[2] A. Y. Yang, Z. Zhou, A. G. Balasubramanian, S. S. Sastry, and Y. Ma, “Fast l1-minimization algorithms for robust face recognition,” *IEEE Trans. Image Process.*, vol. 22, no. 8, pp. 3234–3246, 2013. \
[3] D. M. Malioutov, M. Cetin, and A. S. Willsky, “Homotopy continuation for sparse signal representation,” *in Proc. IEEE Int. Conf. Acoust., Speech, and Signal Process. (ICASSP)*, vol. 5, 2005, pp. 733–736. \
[4]  L. Zhang, M. Yang, and X. Feng, “Sparse representation or collaborative representation: Which helps face recognition?” *in Proc. IEEE Int. Conf. Comput. Vision (ICCV)*, 2011, pp. 471–478. \
[5] K. Kulkarni, S. Lohit, P. Turaga, R. Kerviche, and A. Ashok, “Reconnet: Non-iterative reconstruction of images from compressively sensed measurements,” *in Proc. IEEE Conf. Comput. Vision and Pattern Recognit. (CVPR)*, 2016, pp. 449–458. \
[6] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” *in Proc. IEEE Conf. Comput. Vision and Pattern Recognit. (CVPR)*, 2016, pp. 770–778. \
[7] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna, “Rethinking the inception architecture for computer vision,” *in Proc. IEEE Conf. Comput. Vision and Pattern Recognit. (CVPR)*, 2016, pp. 2818–2826.