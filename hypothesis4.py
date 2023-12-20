import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from libpysal.weights import lat2W  # pip install libpysal
from esda.moran import Moran  # pip install esda

# Load in the MB database for the found and the nasa database for the fallen
df = pd.read_csv(r"Meteoritical Bulletin Database/MB_meteorite_data.csv", sep="|")
df_nasa = pd.read_csv(r"NASA database/Meteorite_Landings.csv", sep=",")

# Exteract the fallen meteorites from nasa database
df_fallen = df_nasa[df_nasa["fall"] == "Fell"].reset_index(drop=True)
df_found = df[df["Fall"] == "Found"].reset_index(drop=True)

# Exteract the found meteorites from MB database
df_fallen = df_fallen[(~df_fallen["reclat"].isna()) | (~df_fallen["reclong"].isna())]
df_found = df_found[(~df_found["Lat"].isna()) | (~df_found["Long"].isna())]


# Set window size for the rolling mean
window = 3

# Set range of years that needs to be analyzed and create dataframe for the fallen meteorites
year_range_fallen = [1850, int(max(df_fallen["year"]))]
fallen_avg = pd.DataFrame(
    columns=[
        "Year",
        "Avg_lat",
        "std_lat",
        "roll_lat_mean",
        "roll_lat_std",
        "detrend_lat",
        "Avg_long",
        "std_long",
        "roll_long_mean",
        "roll_long_std",
        "detrend_long",
    ]
)

# For each year, if any meteorites in that year, add the year & average of the lat/long values of the meteorites
for i in range(int(year_range_fallen[0]), int(year_range_fallen[1]) + 1):
    if i in df_fallen["year"].values:
        df_year = df_fallen[df_fallen["year"] == i]
        fallen_avg.loc[len(fallen_avg.index)] = [
            i,
            np.mean(df_year["reclat"]),
            np.std(df_year["reclat"]),
            np.nan,
            np.nan,
            np.nan,
            np.mean(df_year["reclong"]),
            np.std(df_year["reclong"]),
            np.nan,
            np.nan,
            np.nan,
        ]

# Add the the rolling mean/std for the average Lat/long to dataframe
fallen_avg["roll_lat_mean"] = fallen_avg["Avg_lat"].rolling(window, center=True).mean()
fallen_avg["roll_lat_std"] = fallen_avg["Avg_lat"].rolling(window, center=True).std()

fallen_avg["roll_long_mean"] = (
    fallen_avg["Avg_long"].rolling(window, center=True).mean()
)
fallen_avg["roll_long_std"] = fallen_avg["Avg_long"].rolling(window, center=True).std()

# Add the the detrend for the average Lat/long to dataframe
fallen_avg["detrend_lat"] = fallen_avg["Avg_lat"] - fallen_avg["roll_lat_mean"]
fallen_avg["detrend_long"] = fallen_avg["Avg_long"] - fallen_avg["roll_long_mean"]


# Set range of years that needs to be analyzed and create dataframe for the found meteorites
year_range_found = [1850, int(max(df_found["Year"]))]
found_avg = pd.DataFrame(
    columns=[
        "Year",
        "Avg_lat",
        "std_lat",
        "roll_lat_mean",
        "roll_lat_std",
        "detrend_lat",
        "Avg_long",
        "std_long",
        "roll_long_mean",
        "roll_long_std",
        "detrend_long",
    ]
)

# Do the same as for the fallen, but now for the found meteorites
for i in range(year_range_found[0], year_range_found[1] + 1):
    if i in df_found["Year"].values:
        df_year = df_found[df_found["Year"] == i]
        found_avg.loc[len(found_avg.index)] = [
            i,
            np.mean(df_year["Lat"]),
            np.std(df_year["Lat"]),
            np.nan,
            np.nan,
            np.nan,
            np.mean(df_year["Long"]),
            np.std(df_year["Long"]),
            np.nan,
            np.nan,
            np.nan,
        ]


found_avg["roll_lat_mean"] = found_avg["Avg_lat"].rolling(window, center=True).mean()
found_avg["roll_lat_std"] = found_avg["Avg_lat"].rolling(window, center=True).std()

found_avg["roll_long_mean"] = found_avg["Avg_long"].rolling(window, center=True).mean()
found_avg["roll_long_std"] = found_avg["Avg_long"].rolling(window, center=True).std()

found_avg["detrend_lat"] = found_avg["Avg_lat"] - found_avg["roll_lat_mean"]
found_avg["detrend_long"] = found_avg["Avg_long"] - found_avg["roll_long_mean"]


# Plot detrended Lat/Long of Fallen
plt.figure(figsize=[10, 6])
plt.plot(
    fallen_avg["Year"],
    fallen_avg["detrend_long"],
    color="blue",
    label="Detrended Avg Long",
)
plt.plot(
    fallen_avg["Year"],
    fallen_avg["detrend_lat"],
    color="red",
    label="Detrended Avg Lat",
)
plt.xlim(year_range_fallen[0], year_range_fallen[1])
plt.xlabel(f"Years - ({year_range_fallen[0]} to {year_range_fallen[1]})", fontsize=16)
plt.ylabel("Detrended Avg Lat/Long values", fontsize=16)
plt.title(
    'Time-series of Detrended Lat/Long values of the "Fallen" Meteorites', fontsize=20
)
plt.legend(fontsize=14)
plt.show()

# Plot original Lat/Long of Fallen
plt.figure(figsize=[10, 6])
plt.plot(fallen_avg["Year"], fallen_avg["Avg_long"], color="blue", label=" Avg Long")
plt.plot(fallen_avg["Year"], fallen_avg["Avg_lat"], color="red", label=" Avg Lat")
plt.xlim(year_range_fallen[0], year_range_fallen[1])
plt.xlabel(f"Years - ({year_range_fallen[0]} to {year_range_fallen[1]})", fontsize=16)
plt.ylabel("Avg Lat/Long values", fontsize=16)
plt.title('Time-series of  Lat/Long values of the "Fallen" Meteorites', fontsize=20)
plt.legend(fontsize=14)
plt.show()

# Plot detrended Lat/Long of Found
plt.figure(figsize=[10, 6])
plt.plot(
    found_avg["Year"],
    found_avg["detrend_long"],
    color="blue",
    label="Detrended Avg Long",
)
plt.plot(
    found_avg["Year"], found_avg["detrend_lat"], color="red", label="Detrended Avg Lat"
)
plt.xlim(year_range_found[0], year_range_found[1])
plt.xlabel(f"Years - ({year_range_found[0]} to {year_range_found[1]})", fontsize=16)
plt.ylabel("Detrended Avg Lat/Long values", fontsize=16)
plt.title(
    'Time-series of Detrended Lat/Long values of the "Found" Meteorites', fontsize=20
)
plt.legend(fontsize=14)
plt.show()

# Plot original Lat/Long of Found
plt.figure(figsize=[10, 6])
plt.plot(found_avg["Year"], found_avg["Avg_long"], color="blue", label="Avg Long")
plt.plot(found_avg["Year"], found_avg["Avg_lat"], color="red", label="Avg Lat")
plt.xlim(year_range_found[0], year_range_found[1])
plt.xlabel(f"Years - ({year_range_found[0]} to {year_range_found[1]})", fontsize=16)
plt.ylabel("Avg Lat/Long values", fontsize=16)
plt.title('Time-series of Lat/Long values of the "Found" Meteorites', fontsize=20)
plt.legend(fontsize=14)
plt.show()


# Initialize lists for the autocorrelation values and lag in years
fall_autocorr_lat = []
fall_autocorr_long = []
found_autocorr_lat = []
found_autocorr_long = []
years = []


# For each lag in years, add the autocorrelation to list
for i in range(80):
    years.append(i)
    fall_autocorr_lat.append(fallen_avg["detrend_lat"].autocorr(lag=i))
    fall_autocorr_long.append(fallen_avg["detrend_long"].autocorr(lag=i))
    found_autocorr_lat.append(found_avg["detrend_lat"].autocorr(lag=i))
    found_autocorr_long.append(found_avg["detrend_long"].autocorr(lag=i))


# Plot the autocorrelations for the Fallen meteories
plt.figure(figsize=[10, 6])
plt.plot(years, fall_autocorr_lat, "o-", color="blue", label="Autocorrelations Long")
plt.plot(years, fall_autocorr_long, "o-", color="red", label="Autocorrelations Lat")

plt.xlabel("Lag (in years)", fontsize=16)
plt.ylabel("Autocorrelation", fontsize=16)
plt.title(
    'Autocorrelation of Detrended Lat/Long values of the "Fallen" Meteorites',
    fontsize=20,
)
plt.legend(fontsize=14)
plt.show()

# Plot the autocorrelations for the Found meteories
plt.figure(figsize=[10, 6])
plt.plot(years, found_autocorr_lat, "o-", color="blue", label="Autocorrelations Long")
plt.plot(years, found_autocorr_long, "o-", color="red", label="Autocorrelations Lat")

plt.xlabel("Lag (in years)", fontsize=16)
plt.ylabel("Autocorrelation", fontsize=16)
plt.title(
    'Autocorrelation of Detrended Lat/Long values of the "Found" Meteorites',
    fontsize=20,
)
plt.legend(fontsize=14)
plt.show()


# Convert the detrended locations dataframe to numpy array and filter out nan-values
fallen_np = fallen_avg[["Avg_long", "Avg_lat"]].to_numpy()
tmp = np.where(np.isin(fallen_np, ["NA", "N/A"]), np.nan, fallen_np).astype(float)
fallen_np = tmp[~np.isnan(tmp).any(axis=1)]

found_np = found_avg[["Avg_long", "Avg_lat"]].to_numpy()
tmp = np.where(np.isin(found_np, ["NA", "N/A"]), np.nan, found_np).astype(float)
found_np = tmp[~np.isnan(tmp).any(axis=1)]

# Calculate the Moran's I for the fallen & found meteorites
w = lat2W(fallen_np.shape[0], fallen_np.shape[1])
mi_fallen = Moran(fallen_np, w)

w = lat2W(found_np.shape[0], found_np.shape[1])
mi_found = Moran(found_np, w)

# Run hypothesis test by calculating the MI-distribution for N shuffled datasets.
N = 10**5

MI_fallen = []
MI_found = []
copy_fallen_np = fallen_np
copy_found_np = found_np

for test in range(N):
    # Shuffling for fallen meteorites
    np.random.shuffle(copy_fallen_np)
    w_test = lat2W(copy_fallen_np.shape[0], copy_fallen_np.shape[1])
    mi_test = Moran(copy_fallen_np, w_test)
    MI_fallen.append(mi_test.I)

    # Shuffling for found meteorites
    np.random.shuffle(copy_found_np)
    w_test = lat2W(copy_found_np.shape[0], copy_found_np.shape[1])
    mi_test = Moran(copy_found_np, w_test)
    MI_found.append(mi_test.I)

# Calculate p-value for the MI of the fallen & found meteorites by fitting normal-distribution and calculating z-score
mu, std = norm.fit(MI_fallen)
zscore = (mi_fallen.I - mu) / std
p_mi_fallen = norm.sf(zscore)

mu, std = norm.fit(MI_found)
zscore = (mi_found.I - mu) / std
p_mi_found = norm.sf(zscore)


# Plot MI distributions
fig = plt.figure(figsize=[12, 6])

count1, _, _ = plt.hist(
    MI_fallen,
    bins=50,
    color="red",
    alpha=0.5,
    label="MI distr. of reshuffled fallen Lat/Long",
)
count2, _, _ = plt.hist(
    MI_found,
    bins=50,
    color="blue",
    alpha=0.5,
    label="MI distr. of reshuffled found Lat/Long",
)

n = max(list(count1) + list(count2))
ylim = n + int(0.3 * n)

plt.vlines(
    mi_fallen.I,
    0,
    ylim,
    color="red",
    linestyle="dashed",
    label="MI of Fallen meteorites",
)
plt.text(
    mi_fallen.I,
    ylim / 2,
    " $MI = %.4f $\n $P_{value} = %.4f$" % (mi_fallen.I, p_mi_fallen),
    fontsize=10,
)

plt.vlines(
    mi_found.I,
    0,
    ylim,
    color="blue",
    linestyle="dashed",
    label="MI of Found meteorites",
)
plt.text(
    mi_found.I,
    ylim / 2,
    " $MI = %.4f $\n $P_{value} = %.4f$" % (mi_found.I, p_mi_found),
    fontsize=10,
)

plt.ylim(0, ylim)
plt.legend()
plt.ylabel("counts", fontsize=16)
plt.xlabel("Moran's I", fontsize=16)
plt.title(
    """Moran's I distribution for $N=10^5$ reshuffled sets of Lat/Long 
    values of the \"Fallen\" & \"Found\" Meteorites.""",
    fontsize=20,
)

plt.show()
# fig.savefig("Morans_I_test.png")

# Print MI and P_value for fallen & found
print(f"\nMoran's I of the fallen meteorite locations: {mi_fallen.I:.4f}")
print(f"The corresponding p-value is: {p_mi_fallen:.4f}\n")

print(f"Moran's I of the found meteorite locations: {mi_found.I:.4f}")
print(f"The corresponding p-value is: {p_mi_found:.4f}")
