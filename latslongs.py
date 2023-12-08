import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Download world countries dataset from Natural Earth
land = gpd.read_file('land/World_Land.shp')

grid = gpd.read_file('land/grid.shp')

# grid.plot()
# plt.show()

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
# # Download world coastlines dataset from Natural Earth
# oceans = gpd.read_file(('ne_110m_ocean/ne_110m_ocean.shp')).to_crs(epsg=4326)

# # Filter data for specific latitudes (e.g., between 30 and 60 degrees)
# lat_min, lat_max = -10, 10
# #bbox = Point(lat_min, lat_max).buffer(1).envelope
# #world_filtered =
# world = world.cx[:, lat_min:lat_max]
# #coastlines_filtered =
# oceans = oceans.cx[:, lat_min:lat_max]

# # Create a bounding box for the latitude range


# # Extract land within the bounding box
# # land = gpd.overlay(world, bbox, how='intersection')
# # water = gpd.overlay(oceans, bbox, how='intersection')

# # Calculate total land area and total water area
# total_land_area = world.area.sum()
# total_water_area = oceans.area.sum()

# # Calculate the ratio of land to water
# land_water_ratio = total_land_area / (total_water_area + total_land_area)

# print(f'Total Land Area: {total_land_area:.2f} square units')
# print(f'Total Water Area: {total_water_area:.2f} square units')
# print(f'Land-to-Water Ratio: {land_water_ratio:.2f}')
