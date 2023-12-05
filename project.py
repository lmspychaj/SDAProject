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

# New Fourier & pandas
import matplotlib.pyplot as plt
import cartopy.crs as ccrs # pip install cartopy
import pandas as pd
import numpy as np

# Load dataset
dataset = pd.read_csv('MB_data_temp.csv', delimiter='|', low_memory=False)
print(dataset.head())
print(dataset['Fall'].unique())

# Splitting the '(Lat, Long)' column into two separate columns 'Lat' and 'Long'
dataset[['Lat', 'Long']] = dataset['(Lat,Long)'].str.strip('()').str.split(',', expand=True)

# Converting the new columns to numeric types
dataset['Lat'] = pd.to_numeric(dataset['Lat'], errors='coerce')
dataset['Long'] = pd.to_numeric(dataset['Long'], errors='coerce')

# Discard incorrect coordinates
dataset = dataset[dataset['Long'] < 180]
dataset = dataset[dataset['Long'] > -180]
dataset = dataset[dataset['Lat'] < 90]
dataset = dataset[dataset['Lat'] > -90]

# Prepare map
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

# Split dataset into fallen and found meteorites
fallen = dataset[(dataset['Fall'] == 'Y') | (dataset['Fall'] == 'Yc')]
found = dataset[(dataset['Fall'] == ' ') | (dataset['Fall'] == 'Nd') | (dataset['Fall'] == 'Np') | (dataset['Fall'] == 'Yp')]

# Plot meteorites on map
#map.plot(dataset['reclong'], dataset['reclat'], 'ro', markersize=3)
ax.plot(found['Long'], found['Lat'], 'ro', markersize=3, label='Found')
ax.plot(fallen['Long'], fallen['Lat'], 'bo', markersize=3, label='Fallen')

# Show map
plt.legend()
plt.show()
plt.clf()

# Extract years
years = fallen['Year']

# Filter years with very little data
years = years[years > 1800]
yearrange = int(years.max() - 1800) # change 1800 to years.min when using full dataset

# Create histogram
histogram = plt.hist(years, bins=yearrange)
plt.show()

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
# You might want to identify peaks manually or use a peak finding algorithm
# In this case, we will use the find_peaks function from scipy.signal
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


# Inverse Fourier Transform
# Import ifft from scipy
from scipy.fft import ifft

# Inverse Fourier Transform
time_domain_reconstructed = ifft(yf)

fig, axs = plt.subplots(2)

# Plot reconstructed signal
axs[0].plot(np.arange(1800, 1800 + len(time_domain_reconstructed.real)), time_domain_reconstructed.real, label="Reconstructed")
axs[0].set_ylabel("Reconstructed Count")
axs[0].legend()

# Plot original signal
axs[1].plot(np.arange(1800, 1800 + N), counts, label="Original", alpha=1)
axs[1].set_xlabel("Year")
axs[1].set_ylabel("Original Count")
axs[1].legend()

plt.suptitle("Original vs Reconstructed Signal")
plt.show()

# Testing for significance
num_permutations = 1000
num_samples = len(counts)

# Store Fourier powers for each permutation
permuted_fourier_powers = np.zeros((num_permutations, num_samples // 2))

# Perform the permutations and Fourier analysis
for i in range(num_permutations):
    shuffled_counts = np.random.permutation(counts)
    fft_result_shuffle = fft(shuffled_counts)
    power = 2.0/N * np.abs(fft_result_shuffle[:N//2])
    permuted_fourier_powers[i, :] = power

# Calculate mean and standard deviation for each frequency component's power
mean_powers = np.mean(permuted_fourier_powers, axis=0)
std_powers = np.std(permuted_fourier_powers, axis=0)

# Calculate the 95% confidence intervals for the power
confidence_interval_95_upper = mean_powers + 1.96 * std_powers
confidence_interval_95_lower = mean_powers - 1.96 * std_powers

# Plot the original power spectrum with the confidence intervals
plt.fill_between(xf, confidence_interval_95_lower, confidence_interval_95_upper, color='gray', alpha=0.5)
plt.plot(xf, power_spectrum, label='Original Power Spectrum')
plt.legend()
plt.xlabel('Frequency (1/year)')
plt.ylabel('Power')
plt.title('95% Confidence Intervals of Shuffled Data Fourier Power')
plt.show()

# Identify significant peaks
significant_peaks = power_spectrum > confidence_interval_95_upper
if significant_peaks.any(): 
    print(significant_peaks)
else:
    print("No significant peaks found.")
