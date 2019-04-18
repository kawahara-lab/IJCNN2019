MATLAB implementation of an IJCNN2019 paper titled "Learning with Coherence Patterns in Multivariate Time-series Data via Dynamic Mode Decomposition".

We write below Support Vector Machine as SVM, Dynamic Mode Decomposition as DMD, 
Principal Component Analisys as PCA, t-distributed Stochastic Neighbor Embedding as tSNE.


## Data
Data used in the paper is UCI Daily and Sports Activities Data Set.

The data set contains 8 person's 19 activities.

For more details, see the original web page below.

https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities

After downloading data, run make_data.m in downloaded folder and put outputs in ./data.

## Library
We use well-known SVM library LIBSVM.

Download it from web page below and replace ./libsvm-3.23 with downloaded one.

https://www.csie.ntu.edu.tw/~cjlin/libsvm/.


## Description
svm_DMD.m, svm_SDMD.m and svm_PCA.m perform classification via SVM.

The input of SVM is distance matrix of data.

The distance matrices in those files are computed by DMD modes, Supervised PCA + DMD modes, PCA modes respectively.

tSNE_DMD.m, tSNE_SDMD.m and tSNE_PCA.m perform visualization of the feature vectors above via tSNE.


## Example
Result of svm_SDMD.m is below

![example1](./examples/fig2.png)

Result of tSNE_SDMD.m is below.

![example1](./examples/fig1.png)


## Licence
This sorce code(except the folder "libsvm-3.23", which is not our developing code. Follow the licence of the original.) is released under the MIT Licence, see LICENSE.


## References
T.Bito, M.Hiraoka, Y.Kawahara, "Learning with Coherence Patterns in Multivariate Time-series Data via Dynamic Mode Decomposition," 
in *Proc.* of The International Joint Conference on Neural Networks 2019(IJCNN2019), 2019, pp. xx-xx.

(URL of the paper is comming soon)

LIBSVM -- A Library for Support Vector Machine

https://www.csie.ntu.edu.tw/~cjlin/libsvm/

UCI Machine Learning Repository 

Daily and Sports Activities Data Set.

https://archive.ics.uci.edu/ml/datasets/daily+and+sports+activities
