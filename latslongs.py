import geopandas as gpd
import matplotlib.pyplot as plt

# Read world and grid data
land = gpd.read_file('land/World_Land.shp')
grid = gpd.read_file('land/grid.shp')

# Calculate the amount of land per grid cell
grid["original_area"] = grid.geometry.area
inter = gpd.overlay(df1=land, df2=grid, how="intersection", keep_geom_type=False)
inter["interarea"] = inter.geometry.area
#Each grid cell can intersect multiple land areas. Calculate sum per grid cell..
inter = inter.groupby(by=["ID","original_area"], as_index=False)["interarea"].sum()

#Calculate the ratio of land
inter["landratio"] = inter.apply(lambda x: x["interarea"]/x["original_area"], axis=1)

#Merge the landratio with the grid data
grid = gpd.pd.merge(left=grid, right=inter[["ID","landratio"]], on="ID", how="left")
grid["landratio"].fillna(0, inplace=True)

grid["lat"] = grid.centroid.y
grid["lon"] = grid.centroid.x

#Calculate the mean landratio per latitude and longitude
lat_landratio = grid.groupby(by="lat", as_index=False)["landratio"].mean()
lon_landratio = grid.groupby(by="lon", as_index=False)["landratio"].mean()
print(lat_landratio)
print(lon_landratio)

#Visualization of the landratio
ax = grid.plot(column="landratio", figsize=(10,5), legend=True, cmap="winter")
land.boundary.plot(ax=ax, edgecolor="red")

plt.show()
