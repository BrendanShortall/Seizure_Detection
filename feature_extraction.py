import numpy as np
import pandas as pd
import scipy
from scipy.signal import welch
import numpy as np
import matplotlib.pyplot as plt


#Function gets the mean power of a specified frequency band. Will be used to calculate power estimations of most common frequency bands
def bandpower(data, sf, band, output = False):
    band = np.asarray(band)
    low, high = band

    # Compute the periodogram (Welch)
    freqs, psd = welch(data, 
                       sf, 
                       nperseg=10000,
                       scaling='density', 
                       axis=0)
    
    # put into a df
    psd = pd.DataFrame(psd, index = freqs)
    
    if output:
        print('Welch Output')
        psd.index.name = 'Hz'
        psd.columns = ['Power']
        print(psd)
    
    # Find closest indices of band in frequency vector
    idx_min = np.argmax(np.round(freqs) > low) - 1
    idx_max = np.argmax(np.round(freqs) > high)
    
    # select frequencies of interest
    psd = psd.iloc[idx_min:idx_max,:]
    
    # get the mean of each channel over all frequencies in the band
    psd = psd.mean()
    
    if output:
        print('\nMean Frequency Band')
        print(psd)
    
    return psd

#Returns df of power densities for frequency bands
def power_measures(data, sample_rate, output=False):
    bandpasses = [[[0.1,4],'power_delta'],
                  [[4,8],'power_theta'],
                  [[8,12],'power_alpha'],
                  [[12,30],'power_beta'],
                  [[30,70],'power_gamma']
                 ]
    
    welch_df = pd.DataFrame()
    for bandpass, freq_name in bandpasses:
        bandpass_data = bandpower(data, sample_rate, bandpass)
        bandpass_data.index = [freq_name]
        
        if welch_df.empty:
            welch_df = bandpass_data

        else:
            welch_df = pd.concat([welch_df, bandpass_data])
        
    welch_df = welch_df.T
    
    if output:
        print(welch_df)
    
    return welch_df

# This function takes in a 2D data matrix of size n x d and a threshold array 
# containing two values that correlate to the positive and negative thresholds
# respectively. It returns a n x 1 array containing the number of local maxima 
# and minima peaks that exceed the threshold for a given sample.

def getSpikes(data, threshold):
    
    # Make sure threshold array is formatted correctly
    if threshold[0] < threshold[1]:
        # If the first threshold is smaller than the second threshold, swap them.
        threshold = threshold[::-1]
        
    # Get the number of samples in the data matrix
    n = len(data)
    
    # Initialize an array to store the number of spikes for each sample
    numSpikes = np.zeros((n,1))
    
    for i in range(n):
        # Find the indices of the local maxima and minima in the i-th sample
        max_ind = scipy.signal.find_peaks(data[i])[0]
        min_ind = scipy.signal.find_peaks(-data[i])[0]
        
        # Check if either max_ind or min_ind is empty and replace them with an empty list
        if len(max_ind) == 0:
            max_ind = []
        if len(min_ind) == 0:
            min_ind = []
        
        # Concatenate max_ind and min_ind into a single array to loop through all peaks
        for ind in np.concatenate((max_ind, min_ind)):
            if data[i][ind] > threshold[0] or data[i][ind] < threshold[1]:
                # If the peak exceeds either threshold, increment the spike count for the sample
                numSpikes[i][0] += 1
    
    # Return the array containing the number of spikes for each sample
    return numSpikes


def feature_extraction(data, sample_rate, threshold):
    n = len(data)
    #Compute band powers
    powers = np.zeros((n,5))
    for i in range(n):
        powers[i] = power_measures(data[i], sample_rate, output=False).to_numpy()
    #Get number of action potentials for each sample
    #Could normalize this by dividing by sample rate
    numSpikes = getSpikes(data, threshold)

    #Get mean amplitudes
    means = np.mean(abs(data), axis=1)
    means = means.reshape(n, 1)

    data_final = np.concatenate((numSpikes,means,powers), axis=1)
    df = pd.DataFrame(data_final)
    #Want to scale the data into Z-scores so that a subset of features isn't valued higher than others.
    #only issue is that we must track the mean and stdev for each feature for new data
    df = df.apply(scipy.stats.zscore)
    return df

def main():
    #Load data
    b1 = pd.read_csv(r'C:\Users\shortallb\Documents\Research\Seizure Detection\baseline.csv', header=None).to_numpy()
    s1 = pd.read_csv(r'C:\Users\shortallb\Documents\Research\Seizure Detection\seizure1.csv', header=None).to_numpy()
    s2 = pd.read_csv(r'C:\Users\shortallb\Documents\Research\Seizure Detection\seizure2.csv', header=None).to_numpy()

    #split data into data and labels
    data = np.concatenate((b1,s1,s2), axis = 0)
    labels = data[:,10000]
    data = np.delete(data, 10000, axis=1)
    sample_rate = 10000 # in hz
   
    #Compute baseline statistics for spike detection
    base_mean = np.mean(b1[:,:-1])
    base_stdev = np.std(b1[:,:-1])
    threshold = [base_mean + 4.25*base_stdev, base_mean - 4.25*base_stdev]

    df = feature_extraction(data, sample_rate, threshold)

    #Convert to CSV and output
    df.to_csv("./data.csv")
    df = pd.DataFrame(labels)
    df.to_csv("./labels.csv")

if __name__ == '__main__':
    main()


#Code found from:     
#https://colab.research.google.com/github/Eldave93/Seizure-Detection-Tutorials/blob/master/02.%20Pre-Processing%20%26%20Feature%20Engineering.ipynb#scrollTo=Oe3h7w3tVbhK