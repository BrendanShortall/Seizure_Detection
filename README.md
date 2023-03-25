# Seizure Detection
This repo contains Python code to detect seizure activity from data collected via implanted electrodes in mice brain.

The data used contains 15 mins of baseline activity and 30 mins of seizure activity sampled at 10KHz. This data is partitioned and labeled into 1s windows for classification. A second set of data was added that contains 60 mins of seizure activity and 15 min of baseline activity sampled at 1KHz - this data is also partioned into 1s windows.  

On the original data:
An LSTM model is created using the Keras Python library, and includes K fold cross validation with K = 5. This code is found in seizure_detection.ipynb. The accuracy obtained on the first pass is 68.9%. This is without any filtering or signal processing. 

Callbacks are included to backup the model following each epoch, to terminate the training process once a certain threshold of change in accuracy is not met, and to output the training progress into a .csv file. 

I felt that an LSTM model was overcomplicating a relatively simple task. I decided to pivot to a ML approach instead.

In feature_extraction.py there is code that extracts normalized features for this data. These features are, 5 welch power density estimates for different frequency bands, the average signal amplitude, and the number of spikes in each sample. These features are normalized by computing the mean and standard deviation for each feature across all samples, and converting the raw input into Z-scores. 

In ml_models.ipynb I conduct both an SVM and KNearestNeighbors algorithm (K=7). The data is randomly split into train (75% of samples) and test (25% of samples). Multiple random states have been trained and tested to ensure solidarity amongst cases.

In both models, I have achieved a test accuracy of 100%. 

Future work could include some signal processing leading into the feature extraction to account for signal artefacts. However, the application for this algorithm is time-dependent so it may not be worth the overhead of doing so (another reason why ML felt like a better approach). I will also continue to develop better instructions for others to test on their own data.
