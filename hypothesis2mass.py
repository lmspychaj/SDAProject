import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap # pip install basemap
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def theoretical_lognorm(mu, sigma, x):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x- mu)**2 / (2 * sigma**2))


# Load dataset
dataset = pd.read_csv('Meteorite_Landings.csv', sep=',')
dataset2 = pd.read_csv('Meteoritical Bulletin Database/MB_meteorite_data.csv', sep='|')

print(dataset.head())

# Discard incorrect coordinates
dataset = dataset[dataset['reclong'] < 180]
dataset = dataset[dataset['reclong'] > -180]
dataset = dataset[dataset['reclat'] < 90]
dataset = dataset[dataset['reclat'] > -90]

dataset2 = dataset2[dataset2['Long'] < 180]
dataset2 = dataset2[dataset2['Long'] > -180]
dataset2 = dataset2[dataset2['Lat'] < 90]
dataset2 = dataset2[dataset2['Lat'] > -90]

# Prepare map
# map = Basemap(projection='cyl')
# map.drawmapboundary(fill_color='w')
# map.drawcoastlines()

# Split dataset into fallen and found meteorites
fallen = dataset[dataset['fall'] == 'Fell']
found = dataset2[dataset2['Fall'] == 'Found']

log_mass_fallen = np.log10(fallen['mass (g)'])
histfall = plt.hist(log_mass_fallen, bins=100, density=True, alpha=0.5, label='Fallen')
fallen_cdf = np.cumsum(histfall[0]) / np.sum(histfall[0])

# plt.xlabel('Mass (log10(g))')
# plt.ylabel('Number of fallen meteorites')
# plt.title('Mass distribution of fallen meteorites')

log_mass_found = np.log10(found['Mass (g)'])
log_mass_found = log_mass_found[log_mass_found > -10]
histfound = plt.hist(log_mass_found, bins=100, density=True, alpha=0.5, label='Found')
found_cdf = np.cumsum(histfound[0]) / np.sum(histfound[0])
plt.xlabel('Mass (log10(g))')
plt.ylabel('Probability density')
plt.title('Mass distribution of fallen and found meteorites')

# plt.show()

fallen_mean_mass = np.mean(log_mass_fallen)
fallen_std_mass = np.std(log_mass_fallen)
found_mean_mass = np.mean(log_mass_found)
found_std_mass = np.std(log_mass_found)
space=np.linspace(-2, 10, 1000)
plt.axvline(x=fallen_mean_mass, color='b', linestyle='--')
plt.axvline(x=found_mean_mass, color='r', linestyle='--')

fit_fallen = theoretical_lognorm(fallen_mean_mass, fallen_std_mass, space)
fit_found = theoretical_lognorm(found_mean_mass, found_std_mass, space)

#plt.plot(space, 1/(found_std_mass * np.sqrt(2 * np.pi)) * np.exp( - (space - found_mean_mass)**2 / (2 * found_std_mass**2) ), color='b', label='(Log) Normal distribution fitted to found meteorites')
plt.plot(space, fit_found, color='r', label='(Log10) Normal distribution fitted to found meteorites')
plt.plot(space, fit_fallen, color='b', label='(Log10) Normal distribution fitted to fallen meteorites')
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

plt.hist(bootstrap_means[-1], bins=20, density=True, alpha=0.7, label='Means of bootstrap samples')
plt.axvline(x=np.percentile(bootstrap_means[-1], 2.5), color='r', linestyle='--', label='95% confidence interval')
plt.axvline(x=np.percentile(bootstrap_means[-1], 97.5), color='r', linestyle='--')
plt.axvline(x=fallen_mean_mass, color='r', label='Mean of fallen meteorites')
plt.title('Confidence interval test using the bootstrapped means of the mass distribution of found meteorites')
plt.legend()
plt.show()
plt.clf()

plt.plot([str(size) for size in bootstrap_sizes], pval_means)
plt.title('p-value of KS test for different bootstrap sample sizes')
plt.xlabel('Bootstrap sample size')
plt.ylabel('p-value')
#plt.xticks()
plt.show()
plt.clf()

plt.plot(found_cdf, label='Found')
plt.plot(fallen_cdf, label='Fallen')
plt.legend()
plt.show()

fit_fallen_dist = np.random.normal(fallen_mean_mass, fallen_std_mass, 10000)
fit_found_dist = np.random.normal(found_mean_mass, found_std_mass, 10000)

fit_fallen_hist = plt.hist(fit_fallen_dist, bins=100, density=True)
plt.show()
plt.clf()
# fit_fallen_cdf = np.cumsum(fit_fallen_hist) / np.sum(fit_fallen_hist)

fit_found_hist = plt.hist(fit_found_dist, bins=100, density=True)
# fit_found_cdf = np.cumsum(fit_found_hist) / np.sum(fit_found_hist)

# plt.plot(fit_found_cdf, label='Fallen')
# plt.plot(found_cdf, label='Found')
# plt.show()

print(ks_2samp(fit_fallen_dist, log_mass_fallen))
print(ks_2samp(fit_fallen_dist, log_mass_found))