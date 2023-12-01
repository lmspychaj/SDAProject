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
map.plot(found['reclong'], found['reclat'], 'ro', markersize=3)
map.plot(fallen['reclong'], fallen['reclat'], 'bo', markersize=3)

plt.show()
plt.clf()

# Extract years
years = fallen['year']

# Filter years with very little data
years = years[years > 1800]
yearrange = int(years.max() - 1800) # change 1800 to years.min when using full dataset

# Create histogram
histogram = plt.hist(years, bins=yearrange)
plt.show()

# Extract counts from histogram
counts = histogram[0]

# Autocorrelation function
def autocorrelation(counts):
    result = np.correlate(counts, counts, mode='full')
    print(result)
    print(result.size)
    return result[round(result.size/2):]

print(autocorrelation(counts))

### WIP ###

###
# Fourier analysis: use shuffled dataset to determine whether any periodicity
# found is significant.
###