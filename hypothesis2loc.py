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



# Split dataset into fallen and found meteorites
fallen = dataset[dataset['fall'] == 'Fell']
found = dataset[dataset['fall'] == 'Found']
# plt.plot(fallen['reclong'], fallen['reclat'], 'ro', markersize=1)
# plt.show()
#plt.clf()

# fallhist = np.histogram2d(-fallen['reclat'], fallen['reclong'], bins=50, range=[[-90, 90], [-180, 180]])
# plt.imshow(fallhist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
# plt.colorbar()
# plt.show()
# plt.clf()

# foundhist = np.histogram2d(-found['reclat'], found['reclong'], bins=50, range=[[-90, 90], [-180, 180]])
# plt.imshow(foundhist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
# plt.colorbar()
# plt.show()
# plt.clf()

fallen_lat_mean = fallen['reclat'].mean()
fallen_long_mean = fallen['reclong'].mean()
found_lat_mean = found['reclat'].mean()
found_long_mean = found['reclong'].mean()

# Prepare map
map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines()

plt.axhline(y=fallen_lat_mean, color='r', linestyle='-')
plt.axvline(x=fallen_long_mean, color='r', linestyle='-')
plt.axhline(y=found_lat_mean, color='b', linestyle='-')
plt.axvline(x=found_long_mean, color='b', linestyle='-')
plt.show()

bootstrap_sizes = [5, 25, 100, 200, 500, 1000]
bootstrap_lat_means = []
bootstrap_long_means = []
pvals_lat_means = []
pvals_long_means = []

for size in bootstrap_sizes:
    bootstrap_lat_means_for_size = []
    bootstrap_long_means_for_size = []
    pvals_lat = []
    pvals_long = []
    for _ in range(100):
        bootstrap_indices = np.random.choice(range(len(found)), size=size, replace=True)
        bootstrap_sample = found.iloc[bootstrap_indices]
        bootstrap_lat_mean = np.mean(bootstrap_sample['reclat'])
        bootstrap_long_mean = np.mean(bootstrap_sample['reclong'])
        bootstrap_lat_means_for_size.append(bootstrap_lat_mean)
        bootstrap_long_means_for_size.append(bootstrap_long_mean)
        _, p_value_lat = ks_2samp(fallen['reclat'], bootstrap_sample['reclat'])
        _, p_value_long = ks_2samp(fallen['reclong'], bootstrap_sample['reclong'])
        pvals_lat.append(p_value_lat)
        pvals_long.append(p_value_long)
    pvals_lat_means.append(np.mean(pvals_lat))
    pvals_long_means.append(np.mean(pvals_long))
    bootstrap_lat_means.append(bootstrap_lat_means_for_size)
    bootstrap_long_means.append(bootstrap_long_means_for_size)

# print(bootstrap_lat_means)
# print(bootstrap_long_means)

hist = np.histogram2d([-lat for lat in bootstrap_lat_means[-1]], bootstrap_long_means[-1], bins=50, range=[[-90, 90], [-180, 180]])
# print(hist[0])
# print(hist[1])
plt.imshow(hist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
plt.axvline(x=np.percentile(bootstrap_long_means[-1], 2.5), color='b', linestyle='--')
plt.axvline(x=np.percentile(bootstrap_long_means[-1], 97.5), color='b', linestyle='--')
plt.axhline(y=np.percentile(bootstrap_lat_means[-1], 2.5), color='b', linestyle='--')
plt.axhline(y=np.percentile(bootstrap_lat_means[-1], 97.5), color='b', linestyle='--')

plt.axhline(y=fallen_lat_mean, color='r', linestyle='-')
plt.axvline(x=fallen_long_mean, color='r', linestyle='-')
plt.show()
plt.clf()

plt.plot([str(size) for size in bootstrap_sizes], pvals_lat_means, label='p-value for latitude for bootstrap sample size')
plt.plot([str(size) for size in bootstrap_sizes], pvals_long_means, label='p-value for longitude for bootstrap sample size')
plt.show()

