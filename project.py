import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import numpy as np

# Load dataset
dataset = pd.read_csv('Meteorite_Landings.csv', sep=',')

print(dataset.head())

# Discard incorrect coordinates
dataset = dataset[dataset['reclong'] < 180]
dataset = dataset[dataset['reclong'] > -180]
dataset = dataset[dataset['reclat'] < 90]
dataset = dataset[dataset['reclat'] > -90]

# Prepare map
map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines()

# Split dataset into fallen and found meteorites
fallen = dataset[dataset['fall'] == 'Fell']
found = dataset[dataset['fall'] == 'Found']

# Plot meteorites on map
#map.plot(dataset['reclong'], dataset['reclat'], 'ro', markersize=3)
# map.plot(found['reclong'], found['reclat'], 'ro', markersize=3)
# map.plot(fallen['reclong'], fallen['reclat'], 'bo', markersize=3)

# plt.show()
plt.clf()

# Extract years
years = fallen['year']

# Filter years with very little data
years = years[years > 1860] # before this date less data was collected
yearrange = int(years.max() - 1860) # change 1860 to years.min or when using full dataset

# Create histogram
histogram = plt.hist(years, bins=yearrange)
plt.show()
plt.clf()

# Extract counts from histogram
counts = histogram[0]

# Autocorrelation function
def autocorrelation(counts):
    result = np.correlate(counts, counts, mode='full')
    print(result)
    print(result.size)
    return result[round(result.size/2):]

real_correlations = autocorrelation(counts)

shuffled_correlations = []

for i in range(1, 100):
    # Shuffle counts
    shuffled = np.random.permutation(counts)

    # Calculate autocorrelation
    shuffled_correlation = autocorrelation(shuffled)

    # Add to list
    shuffled_correlations.append(shuffled_correlation)

# Calculate mean and standard deviation of shuffled data
shuffled_correlations = np.array(shuffled_correlations)
mean = np.mean(shuffled_correlations, axis=0)
std = np.std(shuffled_correlations, axis=0)

# Plot 95% confidence interval of shuffled data
plt.fill_between(range(mean.size), mean-std * 2, mean+std * 2, alpha=0.5, label='95% confidence interval of shuffled data')

# Plot real autocorrelation
plt.plot(real_correlations, 'r', label='Autocorrelation of real data')
plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.legend()
plt.show()

### WIP ###

###
# Fourier analysis: use shuffled dataset to determine whether any periodicity
# found is significant.
###

# Fourier Analysis
from scipy.fft import fft, fftfreq

# Number of sample points
N = yearrange
# Sample spacing
T = 1.0  # one year

# Compute the Fast Fourier Transform (FFT)
yf = fft(counts)
xf = fftfreq(N, T)[:N//2]

# Compute the Power Spectrum
power_spectrum = 2.0/N * np.abs(yf[:N//2])

# Plotting the Power Spectrum
plt.plot(xf, power_spectrum)
plt.title("Power Spectrum of Meteor Landings")
plt.xlabel("Frequency (1/year)")
plt.ylabel("Power")
plt.grid()
plt.show()

# Identifying Peaks
from scipy.signal import find_peaks

# Find peaks
peaks, _ = find_peaks(power_spectrum, height=0.1)

# Plotting the Power Spectrum with Peaks
plt.plot(xf, power_spectrum)
plt.plot(xf[peaks], power_spectrum[peaks], "x")
plt.title("Power Spectrum of Meteor Landings")
plt.xlabel("Frequency (1/year)")
plt.ylabel("Power")
plt.grid()
plt.show()


# Import ifft from scipy
from scipy.fft import ifft

# Inverse Fourier Transform
time_domain_reconstructed = ifft(yf)

# Plotting the reconstructed time-domain signal
# Adjust the range of np.arange() to match the length of time_domain_reconstructed.real
plt.plot(np.arange(1800, 1800 + len(time_domain_reconstructed.real)), time_domain_reconstructed.real, label="Reconstructed")
# plt.plot(np.arange(1800, 1800 + N), time_domain_reconstructed.real, label="Reconstructed")
plt.plot(np.arange(1800, 1800 + N), counts, label="Original", alpha=0.5)
plt.legend()
plt.xlabel("Year")
plt.ylabel("Meteor Landings Count")
plt.title("Original vs Reconstructed Signal")
plt.grid()
plt.show()
