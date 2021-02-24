Tensorflow and Keras need to be installed to run this code  
The dataset used for this project was the FLAME dataset using the Training and Test sets  
This datasets can be accesed by affiliates of Essex University using   
https://essexuniversity-my.sharepoint.com/:f:/g/personal/hr17576_essex_ac_uk/EplQh6rwA8pJhHP0jKfg6-kBVHyb1BE9TCAj4MVR0tyOEA

The "directory" variable needs to be changed to the location of the Training Folder  

The DataProcessing code loads the data, divides it into Training and Validation   
and displays the output of loading, applying data augmentation (Random Flips and Random Rotation) to a frame and to the complete dataset  

The ImageDataGenerator code loads the datasets and generate random batches from the initial dataset and applying data augmentation (Brightness, Shear, Horizontal Flip, Zoom,  to create new frames

Both preprocessings can be applied to train a model using .fit and .fit_generator respectively 
