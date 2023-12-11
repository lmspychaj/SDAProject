import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"Meteoritical Bulletin Database\MB_meteorite_data.csv", sep="|")

# Seperate dataframe in fallen and found meteorites
df_fallen = df[df["Fall"] == "Fell"].reset_index(drop=True)
df_found = df[df["Fall"] == "Found"].reset_index(drop=True)

# Obtain all rows with valid long and lat values
df_fallen = df_fallen[(~df_fallen["Lat"].isna()) | (~df_fallen["Long"].isna())]
df_found = df_found[(~df_found["Lat"].isna()) | (~df_found["Long"].isna())]

# window parameter for rolling function
window = 3

# Range of years to analyse
year_range_found = [1850, max(df_found["Year"])]
year_range_fallen = [1850, max(df_fallen["Year"])]


# Calculate avg long/lat and corresponding rolling mean, std and detrend for the fallen meteorites dataframe
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

for i in range(year_range_fallen[0], year_range_fallen[1] + 1):
    if i in df_fallen["Year"].values:
        df_year = df_fallen[df_fallen["Year"] == i]
        fallen_avg.loc[len(fallen_avg.index)] = [
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


fallen_avg["roll_lat_mean"] = fallen_avg["Avg_lat"].rolling(window, center=True).mean()
fallen_avg["roll_lat_std"] = fallen_avg["Avg_lat"].rolling(window, center=True).std()

fallen_avg["roll_long_mean"] = (
    fallen_avg["Avg_long"].rolling(window, center=True).mean()
)
fallen_avg["roll_long_std"] = fallen_avg["Avg_long"].rolling(window, center=True).std()

fallen_avg["detrend_lat"] = fallen_avg["Avg_lat"] - fallen_avg["roll_lat_mean"]
fallen_avg["detrend_long"] = fallen_avg["Avg_long"] - fallen_avg["roll_long_mean"]


# Calculate avg long/lat and corresponding rolling mean, std and detrend for the found meteorites dataframe
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


# Plot for fallen
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

# Plot for found
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


# Calculate autocorrelations of the lon/lat for both fallen and found meteorites df's
fall_autocorr_lat = []
fall_autocorr_long = []
found_autocorr_lat = []
found_autocorr_long = []
years = []
for i in range(25):
    years.append(i)
    fall_autocorr_lat.append(fallen_avg["detrend_lat"].autocorr(lag=i))
    fall_autocorr_long.append(fallen_avg["detrend_long"].autocorr(lag=i))
    found_autocorr_lat.append(found_avg["detrend_lat"].autocorr(lag=i))
    found_autocorr_long.append(found_avg["detrend_long"].autocorr(lag=i))

plt.figure(figsize=[10, 6])
plt.plot(years, fall_autocorr_lat, "o-", color="blue", label="Autocorrelations Long")
plt.plot(years, fall_autocorr_long, "o-", color="red", label="Autocorrelations Lat")
plt.xticks(years)
plt.xlabel(f"Years", fontsize=16)
plt.ylabel("Detrended Avg Lat/Long values", fontsize=16)
plt.title(
    'Time-series of Detrended Lat/Long values of the "Found" Meteorites', fontsize=20
)
plt.legend(fontsize=14)
plt.show()

plt.figure(figsize=[10, 6])
plt.plot(years, found_autocorr_lat, "o-", color="blue", label="Autocorrelations Long")
plt.plot(years, found_autocorr_long, "o-", color="red", label="Autocorrelations Lat")
plt.xticks(years)
plt.xlabel(f"Years", fontsize=16)
plt.ylabel("Detrended Avg Lat/Long values", fontsize=16)
plt.title(
    'Time-series of Detrended Lat/Long values of the "Found" Meteorites', fontsize=20
)
plt.legend(fontsize=14)
plt.show()
