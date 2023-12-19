import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def theoretical_lognorm(mu, sigma, x):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x- mu)**2 / (2 * sigma**2))


# Load dataset
dataset = pd.read_csv('NASA database/Meteorite_Landings.csv', sep=',')
dataset2 = pd.read_csv('Meteoritical Bulletin Database/MB_meteorite_data.csv', sep='|')

# Discard incorrect coordinates
dataset = dataset[dataset['reclong'] < 180]
dataset = dataset[dataset['reclong'] > -180]
dataset = dataset[dataset['reclat'] < 90]
dataset = dataset[dataset['reclat'] > -90]

dataset2 = dataset2[dataset2['Long'] < 180]
dataset2 = dataset2[dataset2['Long'] > -180]
dataset2 = dataset2[dataset2['Lat'] < 90]
dataset2 = dataset2[dataset2['Lat'] > -90]

# Split into fallen and found meteorites
fallen = dataset[dataset['fall'] == 'Fell']
found = dataset2[dataset2['Fall'] == 'Found']

# Extract the logarithm of the meteorite masses. Plot the histograms and create CDFs
log_mass_fallen = np.log10(fallen['mass (g)'])
histfall = plt.hist(log_mass_fallen, bins=100, density=True, alpha=0.5, label='Fallen')
fallen_cdf = np.cumsum(histfall[0]) / np.sum(histfall[0])

log_mass_found = np.log10(found['Mass (g)'])
log_mass_found = log_mass_found[log_mass_found > -10]
histfound = plt.hist(log_mass_found, bins=100, density=True, alpha=0.5, label='Found')
found_cdf = np.cumsum(histfound[0]) / np.sum(histfound[0])
plt.xlabel('Mass (log10(g))')
plt.ylabel('Probability density')
plt.title('Mass distribution of fallen and found meteorites')

# Determine the mean and standard deviation of the mass distributions. Add a vertical line for the means.
# Fit a normal distribution to the mass distributions and plot the fitted distributions.
fallen_mean_mass = np.mean(log_mass_fallen)
fallen_std_mass = np.std(log_mass_fallen)
found_mean_mass = np.mean(log_mass_found)
found_std_mass = np.std(log_mass_found)
space=np.linspace(-2, 10, 1000)
plt.axvline(x=fallen_mean_mass, color='b', linestyle='--')
plt.axvline(x=found_mean_mass, color='r', linestyle='--')

fit_fallen = theoretical_lognorm(fallen_mean_mass, fallen_std_mass, space)
fit_found = theoretical_lognorm(found_mean_mass, found_std_mass, space)

# Plot fitted normal distribution
plt.plot(space, fit_found, color='r', label='(Log10) Normal distribution fitted to found meteorites')
plt.plot(space, fit_fallen, color='b', label='(Log10) Normal distribution fitted to fallen meteorites')
plt.legend()
plt.show()
plt.clf()

# Different bootstrap sample sizes
bootstrap_sizes = [5, 25, 100, 200, 500, len(log_mass_found)]
bootstrap_means = []
pval_means = []

for size in bootstrap_sizes:
    bootstrap_means_for_size = []
    pvals = []
    # Calculate bootstrap means and p-values for each bootstrap sample size
    for _ in range(100):
        bootstrap_sample = np.random.choice(log_mass_found, size=size, replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means_for_size.append(bootstrap_mean)
        _, p_value = ks_2samp(log_mass_fallen, bootstrap_sample)
        pvals.append(p_value)
    pval_means.append(np.mean(pvals))
    bootstrap_means.append(bootstrap_means_for_size)

# For demonstration, plot the histograms of the bootstrap means and the 95% confidence interval for the largest sample size.
plt.hist(bootstrap_means[-1], bins=20, density=True, alpha=0.7, label='Means of bootstrap samples')
plt.axvline(x=np.percentile(bootstrap_means[-1], 2.5), color='r', linestyle='--', label='95% confidence interval')
plt.axvline(x=np.percentile(bootstrap_means[-1], 97.5), color='r', linestyle='--')
plt.axvline(x=fallen_mean_mass, color='r', label='Mean of fallen meteorites')
plt.title('Confidence interval test using the bootstrapped means of the mass distribution of found meteorites')
plt.legend()
plt.show()
plt.clf()

# Plot the average p-values for the different bootstrap sample sizes
plt.plot([str(size) for size in bootstrap_sizes], pval_means)
plt.title('p-value of KS test for different bootstrap sample sizes')
plt.xlabel('Bootstrap sample size')
plt.ylabel('p-value')
plt.show()
plt.clf()

# Create histograms and cdfs of the fitted distributions
fit_fallen_dist = np.random.normal(fallen_mean_mass, fallen_std_mass, 5000)
fit_found_dist = np.random.normal(found_mean_mass, found_std_mass, 5000)

fit_fallen_hist, _ = np.histogram(fit_fallen_dist, bins=100, density=True)
fit_fallen_cdf = np.cumsum(fit_fallen_hist) / np.sum(fit_fallen_hist)

fit_found_hist, _ = np.histogram(fit_found_dist, bins=100, density=True)
fit_found_cdf = np.cumsum(fit_found_hist) / np.sum(fit_found_hist)

# Plot the cdfs of the fitted distributions and the original distributions
plt.subplot(1, 2, 1)
plt.title('CDFs of fitted and original fallen distribution')
plt.plot(fit_fallen_cdf, label='Fitted CDF')
plt.plot(fallen_cdf, label='CDF')
plt.legend()
plt.subplot(1, 2, 2)
plt.title('CDFs of fitted and original found distribution')
plt.plot(fit_found_cdf, label='Fitted CDF')
plt.plot(found_cdf, label='CDF')
plt.legend()
plt.show()

# Run KS test on the fitted distributions, comparing them to the real distributions
pvals_fallen = []
pvals_found = []

for _ in range(200):
    fit_fallen_dist = np.random.normal(fallen_mean_mass, fallen_std_mass, 5000)
    fit_found_dist = np.random.normal(found_mean_mass, found_std_mass, 5000)

    pvals_fallen.append(ks_2samp(fit_fallen_dist, log_mass_fallen)[1])
    pvals_found.append(ks_2samp(fit_found_dist, log_mass_found)[1])

print("p-value of KS test for fallen distribution: ", np.mean(pvals_fallen))
print("p-value of KS test for found distribution: ", np.mean(pvals_found))

print("This means the mass of the fallen meteorites follows the distribution with mean", fallen_mean_mass, "and standard deviation", fallen_std_mass)