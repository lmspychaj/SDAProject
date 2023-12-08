import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# Load dataset
dataset = pd.read_csv('Meteorite_Landings.csv', sep=',')

print(dataset.head())

# Discard incorrect coordinates
dataset = dataset[dataset['GeoLocation'] != "(0.0, 0.0)"]
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
# plt.plot(fallen['reclong'], fallen['reclat'], 'ro', markersize=1)
# plt.show()
# plt.clf()

# fallhist = np.histogram2d(-fallen['reclat'], fallen['reclong'], bins=50)
# plt.imshow(fallhist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
# plt.colorbar()
# plt.show()
# plt.clf()

# foundhist = np.histogram2d(-found['reclat'], found['reclong'], bins=50)
# plt.imshow(foundhist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
# plt.colorbar()
# plt.show()
# plt.clf()

fallen_lat_mean = fallen['reclat'].mean()
fallen_long_mean = fallen['reclong'].mean()
found_lat_mean = found['reclat'].mean()
found_long_mean = found['reclong'].mean()

plt.axhline(y=fallen_lat_mean, color='r', linestyle='-')
plt.axvline(x=fallen_long_mean, color='r', linestyle='-')
plt.axhline(y=found_lat_mean, color='b', linestyle='-')
plt.axvline(x=found_long_mean, color='b', linestyle='-')
plt.show()

bootstrap_lat_means = []
bootstrap_long_means = []
for _ in range(100):
    bootstrap_indices = np.random.choice(range(len(found)), size=1000, replace=True)
    bootstrap_sample = found.iloc[bootstrap_indices]
    bootstrap_lat_mean = np.mean(bootstrap_sample['reclat'])
    bootstrap_long_mean = np.mean(bootstrap_sample['reclong'])
    bootstrap_lat_means.append(bootstrap_lat_mean)
    bootstrap_long_means.append(bootstrap_long_mean)

hist = np.histogram2d(bootstrap_lat_means, bootstrap_long_means, bins=50)
plt.imshow(hist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
plt.axvline(x=np.percentile(bootstrap_long_means, 2.5), color='b', linestyle='--')
plt.axvline(x=np.percentile(bootstrap_long_means, 97.5), color='b', linestyle='--')
plt.axhline(y=np.percentile(bootstrap_lat_means, 2.5), color='b', linestyle='--')
plt.axhline(y=np.percentile(bootstrap_lat_means, 97.5), color='b', linestyle='--')

plt.axhline(y=fallen_lat_mean, color='r', linestyle='-')
plt.axvline(x=fallen_long_mean, color='r', linestyle='-')
plt.show()


