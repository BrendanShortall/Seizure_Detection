# Seizure Detection
This repo contains Python code to detect seizure activity from data collected via implanted electrodes in mice brain.

The data used contains 15 mins of baseline activity and 30 mins of seizure activity sampled at 10KHz. This data is partitioned and labeled into 1s windows for classification.

An LSTM model is created using the Keras Python library, and includes K fold cross validation with K = 5. The accuracy obtained on the first pass is 68.9%.

Callbacks are included to backup the model following each epoch, to terminate the training process once a certain threshold of change in accuracy is not met, and to output the training progress into a .csv file. 

More work will be done for preprocessing, likely an IIR bandpass filter for noise and consideration of smaller window sizes. I am also exploring feature extraction methods for a machine learning model. So far, I have welch power density estimates for each of the main classes of brain wave frequencies and have also explored the PyEEG library for other common statistical features for EEG classification. Future work will also be done to fine tune the LSTM hyperparameters in an attempt to maximize the accuracy. 
