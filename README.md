# 工件裂痕偵測
## Target：segment the crack on metal component with high accuracy
This project 
### Function Included
  - data preprocessing  
  - dataset build  
  - Model design  
  - Model Build  
  - Training/Evaluate/Predict
### Reference 
  - Python 3.10
  - WSL platform (Ubuntu)
  - virtual environment
  - visual studio code
  - CUDA toolkit 12.4  
  - Tensorflow 2.16.1  
### Explaination  
  - data.py：use for data/label preprocessing and Create tensorflow Dataset  
  - model.py：definition of model structure class  
  - train.py：training/saving model, also use for generating chart of evaluation (Mainly Execute)  
  - predict.py：load '.keras' file to use trained model for predicting (Mainly Execute)
### Tutorial  
  - guide 1. create dataset
  1. check the file root of data and label
  2. use *DScreate* in data.py or import it in other .py
  ![螢幕擷取畫面 2024-05-31 164648](https://github.com/telun0130/tfProject/assets/145544962/448e435a-e533-4726-aba0-c971f0cc5435)
  3. get the tf Dataset as return
  - guide 2. build model
  1. find the model in *ModelSet* of model.py (we have U-Net, R2U-Net now)
  2. use *ModelCreator* in model.py or import it, input the model name and get the model as return
  ![image](https://github.com/telun0130/tfProject/assets/145544962/1f6ef807-1e25-4a8e-a5a5-74fa64b6912f)
  - guide 3. Training
  1. call the *trainer* class in train.py
  2. use the function (train, trainwithsave) in trainer and input the parameter(dataset, epoch, input/output channel...)
  3. start training
  ![image](https://github.com/telun0130/tfProject/assets/145544962/d447857b-4c53-427a-8352-a5a96a9b4b0e)
  4. get the evaluation plot
  - guide 4. Predicting
  1. call the *predictor* in predict.py
  2. load the model by input the path of .keras file
  3. input data to predict  
  ![image](https://github.com/telun0130/tfProject/assets/145544962/f65e9892-dbc9-49e1-b513-2ca2cf22e2ff)

