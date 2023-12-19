import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap # pip install basemap
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# Load dataset
dataset = pd.read_csv('Meteorite_Landings.csv', sep=',')
dataset2 = pd.read_csv('Meteoritical Bulletin Database/MB_meteorite_data.csv', sep='|')

# Discard incorrect coordinates
dataset = dataset[dataset['GeoLocation'] != "(0.0, 0.0)"]
dataset = dataset[dataset['reclong'] < 180]
dataset = dataset[dataset['reclong'] > -180]
dataset = dataset[dataset['reclat'] < 90]
dataset = dataset[dataset['reclat'] > -90]

dataset2 = dataset2[dataset2['Long'] < 180]
dataset2 = dataset2[dataset2['Long'] > -180]
dataset2 = dataset2[dataset2['Lat'] < 90]
dataset2 = dataset2[dataset2['Lat'] > -90]
dataset2 = dataset2[dataset2['GeoLocation'] != ""]



# Split into fallen and found meteorites. Plot the locations of the meteorites.
fallen = dataset[dataset['fall'] == 'Fell']
found = dataset2[dataset2['Fall'] == 'Found']
plt.plot(fallen['reclong'], fallen['reclat'], 'ro', markersize=1)
plt.show()
plt.clf()

# Plot a histogram of the locations of fallen meteorites.
fallhist = np.histogram2d(-fallen['reclat'], fallen['reclong'], bins=50, range=[[-90, 90], [-180, 180]])
plt.imshow(fallhist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
plt.colorbar()
plt.title('Location of fallen meteorites')
plt.show()
plt.clf()

# Plot a histogram of the locations of found meteorites.
foundhist = np.histogram2d(-found['Lat'], found['Long'], bins=50, range=[[-90, 90], [-180, 180]])
plt.imshow(foundhist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
plt.colorbar()
plt.title('Location of found meteorites')
plt.show()
plt.clf()

# Determine the mean latitude and longitude of the fallen and found meteorites. Plot the means on a map.
fallen_lat_mean = fallen['reclat'].mean()
fallen_long_mean = fallen['reclong'].mean()
found_lat_mean = found['Lat'].mean()
found_long_mean = found['Long'].mean()

# Prepare map
map = Basemap(projection='cyl')
map.drawmapboundary(fill_color='w')
map.drawcoastlines()

plt.axhline(y=fallen_lat_mean, color='b', linestyle='-', label='Mean location of fallen meteorites')
plt.axvline(x=fallen_long_mean, color='b', linestyle='-')
plt.axhline(y=found_lat_mean, color='r', linestyle='-', label='Mean location of found meteorites')
plt.axvline(x=found_long_mean, color='r', linestyle='-')
plt.legend()
plt.show()

# Perform a Kolmogorov-Smirnov test to determine whether the latitude and longitude
# of the fallen and found meteorites come from the same distribution.
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
        bootstrap_lat_mean = np.mean(bootstrap_sample['Lat'])
        bootstrap_long_mean = np.mean(bootstrap_sample['Long'])
        bootstrap_lat_means_for_size.append(bootstrap_lat_mean)
        bootstrap_long_means_for_size.append(bootstrap_long_mean)
        _, p_value_lat = ks_2samp(fallen['reclat'], bootstrap_sample['Lat'])
        _, p_value_long = ks_2samp(fallen['reclong'], bootstrap_sample['Long'])
        pvals_lat.append(p_value_lat)
        pvals_long.append(p_value_long)
    pvals_lat_means.append(np.mean(pvals_lat))
    pvals_long_means.append(np.mean(pvals_long))
    bootstrap_lat_means.append(bootstrap_lat_means_for_size)
    bootstrap_long_means.append(bootstrap_long_means_for_size)

# Plot the histogram of the bootstrap means and the mean locations of the fallen meteorites.
hist = np.histogram2d([-lat for lat in bootstrap_lat_means[-1]], bootstrap_long_means[-1], bins=50, range=[[-90, 90], [-180, 180]])
plt.imshow(hist[0], extent=[-180, 180, -90, 90], norm=mpl.colors.LogNorm(), cmap='Greens')
plt.axvline(x=np.percentile(bootstrap_long_means[-1], 2.5), color='r', linestyle='--', label='95% confidence interval of found meteorite means')
plt.axvline(x=np.percentile(bootstrap_long_means[-1], 97.5), color='r', linestyle='--')
plt.axhline(y=np.percentile(bootstrap_lat_means[-1], 2.5), color='r', linestyle='--')
plt.axhline(y=np.percentile(bootstrap_lat_means[-1], 97.5), color='r', linestyle='--')

plt.axhline(y=fallen_lat_mean, color='b', linestyle='-' , label='Mean location of fallen meteorites')
plt.axvline(x=fallen_long_mean, color='b', linestyle='-')
plt.legend()
plt.show()
plt.clf()

# Plot the average p-values for the different bootstrap sample sizes for latitude and longitude.
plt.plot([str(size) for size in bootstrap_sizes], pvals_lat_means, label='p-value for latitude for bootstrap sample size')
plt.plot([str(size) for size in bootstrap_sizes], pvals_long_means, label='p-value for longitude for bootstrap sample size')
plt.title('p-values for latitude and longitude for different bootstrap sample sizes')
plt.legend()
plt.show()

