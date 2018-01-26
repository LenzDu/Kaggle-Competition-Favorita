# Kaggle-Competition-Favorita

This is 5th place solution for Kaggle competition Favorita Grocery Sales Forecasting.  

## The Problem

This competition is a time series problem where we are required to predict the sales of different items in different stores for 16 days in the future, given the sales history and promotion info of these items. Additional information about the items and the stores are also provided. Dataset and detailed description can be found on the competition page: https://www.kaggle.com/c/favorita-grocery-sales-forecasting

## Model Overview

I build 3 models: a Gradient Boosting, a CNN+DNN and a seq2seq RNN model. Final model was a weighted average of these models (where each model is stabilized by training multiple times with different random seeds then take the average). Each model separately can stay in top 1% in the final ranking.

**LGBM:** It is an upgraded model from the public kernels. More features, data and periods were fed to the model.

**CNN+DNN:** This is a traditional NN model, where the CNN part is a dilated causal convolution inspired by WaveNet, and the DNN part is 2 FC layers connected to raw sales sequences. Then the inputs are concatenated together with categorical embeddings and future promotions, and directly output to 16 future days of predictions.

**RNN:** This is a seq2seq model with a similar architecture of @Arthur Suilin's solution for the web traffic prediction. Encoder and decoder are both GRUs. The hidden states of the encoder are passed to the decoder through an FC layer connector. This is useful to improve the accuracy significantly.

## How to Run the Model

Three models are in separate .py files as their filename tell.

Before running the models, use the function *load_data()* in Utils.py to load and transform the raw data files, and use *save_unstack()* to save them to feather files. In the model codes, change the input of *load_unstack()* to the filename you saved. Then the models can be runned. Please read the codes of these functions for more details.

Note: if you are not using a GPU, change CudnnGRU to GRU in seq2seq.py
