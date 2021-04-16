**Wildfire detection**


**Instructions**   
This solution is divided into two approaches:  
1. Train a model only on FLAME dataset (Specific Model)
2. Train a model first on FireNet (General Model) and then train it on FLAME (General/Specific)

The same model preparation was performed on both approaches, using Transfer Learning with Xception loading ImageNet weights, and then perform
finetunning on the last layers

A Revised Dataset was produced from the original FLAME [1] dataset, this Revised dataset was used to train and test the models used on this work, it can be accessed here
https://drive.google.com/drive/folders/1hfiougCgdAEfuYx9JhdKQMYJd0u70pzA?usp=sharing  
Download both the Train_Revised and Test_Revised and store their path to use it as input for the models  
  
  
The other dataset used was FireNet [2] only utilizng the Training Dataset.zip https://drive.google.com/drive/folders/1HznoBFEd6yjaLFlSmkUGARwCUzzG4whq?usp=sharing
, download the zip and store its path for the General/Specific model, the program will unzip and reorganize the data automatically  
*Some of the frames could not be loaded correctly and are leaved when preparing the FireNet dataset*

Once the files have been prepared the code can be runned 

The codes are divided in:  
*Model Training which included preparing the dataset, the model and fiting the model  
*Testing to observe directly how the model performs by loading it and preparing only the Test dataset  

*General/Specific trains the model first on FireNet and then on FLAME   
*Specific only trains the model on FLAME  


All models produced on this approach can be accesed here:   
https://drive.google.com/drive/folders/1hjLq7FhpuQuNNtRXjlGuBV7_OtT4c95-?usp=sharing

The models can be loaded directly for quicker analysis on their performance without the need to retrain them, only need to change the path on the Testing part on 
model.load to the desired model path, and prepare the Test Set

*  
*  
*  
*  
[1] Alireza Shamsoshoara, Fatemeh Afghah, Abolfazl Razi, Liming Zheng, Peter Fulé, Erik Blasch, November 19, 2020, "The FLAME dataset: Aerial Imagery Pile burn detection using drones (UAVs)", IEEE Dataport, doi: https://dx.doi.org/10.21227/qad6-r683 .

[2]  Jadon, Arpit, et al. "FireNet: a specialized lightweight fire & smoke detection model for real-time IoT applications." arXiv preprint arXiv:1905.11922 (2019). 
