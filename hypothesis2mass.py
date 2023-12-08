import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# Load dataset
dataset = pd.read_csv('Meteorite_Landings.csv', sep=',')

print(dataset.head())

# Discard incorrect coordinates
dataset = dataset[dataset['reclong'] < 180]
dataset = dataset[dataset['reclong'] > -180]
dataset = dataset[dataset['reclat'] < 90]
dataset = dataset[dataset['reclat'] > -90]

# Prepare map
# map = Basemap(projection='cyl')
# map.drawmapboundary(fill_color='w')
# map.drawcoastlines()

# Split dataset into fallen and found meteorites
fallen = dataset[dataset['fall'] == 'Fell']
found = dataset[dataset['fall'] == 'Found']

log_mass_fallen = np.log10(fallen['mass (g)'])
plt.hist(log_mass_fallen, bins=100, density=True, alpha=0.5, label='Fallen')
# plt.xlabel('Mass (log10(g))')
# plt.ylabel('Number of fallen meteorites')
# plt.title('Mass distribution of fallen meteorites')

log_mass_found = np.log10(found['mass (g)'])
log_mass_found = log_mass_found[log_mass_found > -10]
plt.hist(log_mass_found, bins=100, density=True, alpha=0.5, label='Found')
plt.xlabel('Mass (log10(g))')
plt.ylabel('Probability density')
plt.title('Mass distribution of fallen and found meteorites')
plt.legend()
# plt.show()

fallen_mean_mass = np.mean(log_mass_fallen)
found_mean_mass = np.mean(log_mass_found)
plt.axvline(x=fallen_mean_mass, color='r', linestyle='--')
plt.axvline(x=found_mean_mass, color='b', linestyle='--')

plt.show()
plt.clf()

bootstrap_means = []
for _ in range(100):
    bootstrap_sample = np.random.choice(log_mass_found, size=len(log_mass_found), replace=True)
    bootstrap_mean = np.mean(bootstrap_sample)
    bootstrap_means.append(bootstrap_mean)

plt.hist(bootstrap_means, bins=20, density=True, alpha=0.5)
plt.axvline(x=np.percentile(bootstrap_means, 2.5), color='r', linestyle='--')
plt.axvline(x=np.percentile(bootstrap_means, 97.5), color='r', linestyle='--')
plt.axvline(x=fallen_mean_mass, color='r')

plt.show()

ks_statistic, p_value = ks_2samp(log_mass_fallen, log_mass_found)

print("The p-value when using the Kolmogorov-Smirnov test is:", p_value)
