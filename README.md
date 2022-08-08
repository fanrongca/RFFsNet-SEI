# RFFsNet-SEI

# Introduction

We propose an accelerated emitter identification approach based on hybrid driven scheme, namely RFFsNet-SEI. As RFFsNet-SEI identifies individual of emitters from received raw data in end-to-end, it accelerates the SEI implementation and simplifies the identification procedures. Based on simulation dataset and real dataset collected in the anechoic chamber, in terms of identification accuracy, computational complexity, and prediction speed, results illustrate that the proposed method outperforms existing methods.

Through steps S1 → S6, in order to illustrate the effectiveness of the proposed neural network, we show the design process of RFFsNet-SEI in an incremental manner.


# Updates
- 【2022/08/08】We upload the source code of RFFsNet-SEI model
- 【2022/08/09】We upload test files and prediction code
  
# Environments

- python 3.8.6

- PyCharm Community 2018.3.2

- CUDA 10.0

- NVIDIA GeForce RTX2080
  
- Two Intel Xeon E5-2678v3 @2.50GHz CPUs and 128GB RAM

# Requirements

- h5py 2.10.0

- numpy 1.19.3
  
- pandas 0.25.0

- tensorflow 1.12.0

- tensorflow-gpu 1.13.1

# File description
- RFFsNet_SEI_train.py -- Data preprocessing, model training, trained model storage function.
- RFFsNet_SEI_predict.py -- Be responsible for reading the trained model and valid file, outputting the predicted classsification results and calculating the Accuracy.
- \10types\Valid.tfrecord -- A demonstration validation file containing sample and lable for 10 simulated PAs.
- \10types\saved_model.pb -- the trained model of RFFsNet-SEI for 10 simulated PAs, "variables" folder contains the trained variables.
- \8types\Valid.tfrecord -- A demonstration validation file containing sample and lable for 8 real PAs.
- \8types\saved_model.pb -- the trained model of RFFsNet-SEI for 8 real PAs, "variables" folder contains the trained variables.

# validation step
- (1) open the RFFsNet_SEI_predict.py
- (2) In Line 19, modify the "saved_model_dir" to the full path where the trained model file (ie. saved_model.pb) is located.
      Noted that saved_model.pb and the "variables" folder should remain in the same level of file path
- (3) In Line 20, modify the "test_path" to the full path of the Valid tfrecord file (ie. Valid.tfrecord) 
- (4) In Line 21, When the measured data set of 8 PAs is selected, "real_num" is set to 8; conversely, when the real data set of 10 PAs is selected, "real_num" is set to 8
- (4) Run the RFFsNet_SEI_predict.py
- (5) The console will print Acc of the prediction
- (6) CSV file (ie. predict.csv) containing real and predicted classification is generated in "saved_model_dir"

# Contact
Issues should be raised directly in the repository. For professional support requests for the current version of the code, please email fanrong@cafuc.edu.cn.
  
  
  
  
  
