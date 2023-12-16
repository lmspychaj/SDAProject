import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Download world countries dataset from Natural Earth
land = gpd.read_file('land/World_Land.shp')

grid = gpd.read_file('land/grid.shp')

grid["original_area"] = grid.geometry.area
print(grid.columns)
inter = gpd.overlay(df1=land, df2=grid, how="intersection", keep_geom_type=False)
inter["interarea"] = inter.geometry.area
#Each grid cell can intersect multiple land areas. Calculate sum per grid cell..
inter = inter.groupby(by=["ID","original_area"], as_index=False)["interarea"].sum()

#Calculate the ratio of land
inter["landratio"] = inter.apply(lambda x: x["interarea"]/x["original_area"], axis=1)

grid = gpd.pd.merge(left=grid, right=inter[["ID","landratio"]], on="ID", how="left")

grid["landratio"].fillna(0, inplace=True)

grid["lat"] = grid.centroid.y
grid["lon"] = grid.centroid.x

lat_landratio = grid.groupby(by="lat", as_index=False)["landratio"].mean()
lon_landratio = grid.groupby(by="lon", as_index=False)["landratio"].mean()
print(lat_landratio)
print(lon_landratio)

ax = grid.plot(column="landratio", figsize=(10,5), legend=True, cmap="winter")
land.boundary.plot(ax=ax, edgecolor="red")

# world.plot()
plt.show()
