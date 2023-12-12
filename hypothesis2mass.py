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

# plt.show()

fallen_mean_mass = np.mean(log_mass_fallen)
fallen_std_mass = np.std(log_mass_fallen)
found_mean_mass = np.mean(log_mass_found)
found_std_mass = np.std(log_mass_found)
space=np.linspace(-2, 10, 1000)
plt.axvline(x=fallen_mean_mass, color='r', linestyle='--')
plt.axvline(x=found_mean_mass, color='b', linestyle='--')

plt.plot(space, 1/(found_std_mass * np.sqrt(2 * np.pi)) * np.exp( - (space - found_mean_mass)**2 / (2 * found_std_mass**2) ), color='b', label='(Log) Normal distribution fitted to found meteorites')
plt.plot(space, 1/(fallen_std_mass * np.sqrt(2 * np.pi)) * np.exp( - (space - fallen_mean_mass)**2 / (2 * fallen_std_mass**2) ), color='r', label='(Log) Normal distribution fitted to fallen meteorites')

plt.legend()
plt.show()
plt.clf()

bootstrap_sizes = [5, 25, 100, 200, 500, len(log_mass_found)]
bootstrap_means = []
pval_means = []

for size in bootstrap_sizes:
    bootstrap_means_for_size = []
    pvals = []
    for _ in range(100):
        bootstrap_sample = np.random.choice(log_mass_found, size=size, replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means_for_size.append(bootstrap_mean)
        _, p_value = ks_2samp(log_mass_fallen, bootstrap_sample)
        pvals.append(p_value)
    pval_means.append(np.mean(pvals))
    bootstrap_means.append(bootstrap_means_for_size)
# for _ in range(100):
#     bootstrap_sample = np.random.choice(log_mass_found, size=len(log_mass_found), replace=True)
#     bootstrap_mean = np.mean(bootstrap_sample)
#     bootstrap_means.append(bootstrap_mean)

plt.hist(bootstrap_means[-1], bins=20, density=True, alpha=0.5)
plt.axvline(x=np.percentile(bootstrap_means[-1], 2.5), color='r', linestyle='--')
plt.axvline(x=np.percentile(bootstrap_means[-1], 97.5), color='r', linestyle='--')
plt.axvline(x=fallen_mean_mass, color='r')

plt.show()
plt.clf()

plt.plot([str(size) for size in bootstrap_sizes], pval_means)
plt.title('p-value of KS test for different bootstrap sample sizes')
plt.xlabel('Bootstrap sample size')
plt.ylabel('p-value')
#plt.xticks()
plt.show()
