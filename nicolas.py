import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as crs

dataPath = "../Data/"

def getIsochrones(dataPath):
	d = xr.open_dataset(dataPath + "isochrone_buffers/PaleoGrIS_1.0_isochrone_buffers_2km_age.nc")
	return d

def getMF(dataPath):
	return xr.open_dataset(dataPath + "GISModel/fixed-seasonality.nc").h

def getMV(dataPath):
	return xr.open_dataset(dataPath + "GISModel/variable-seasonality.nc").h


def getIsochronesOld(dataPath, name):
	for i, confidence in enumerate(["very_low", "low", "mid", "high"]):
		try:
			dd = xr.open_dataset(dataPath + f"Isochrones/{name}_ka_{confidence}_confidence_isochrone.nc").isochrone
			if "d" not in locals():
				d = dd
			else:
				d = d + dd*(i+1)
		except:
			print(f"Isochrones {name} does not have confidence {confidence}")
	ll = xr.open_dataset(dataPath + "Isochrones/lat_lon.nc")
	print(d)
	# d["lat"] = ll.lat
	print(d)
	exit()
	return d


def plot(time):
	fix = getMF(dataPath)
	var = getMV(dataPath)
	iso = getIsochrones(dataPath)
	print(iso.isochrone.values.tolist())

	fig = plt.figure(figsize=(16,9), constrained_layout=True)
	gs = fig.add_gridspec(1, 3)
	ax1 = fig.add_subplot(gs[0,0], projection=crs.NorthPolarStereo(central_longitude=360-45))
	ax1.pcolormesh(fix.x1, fix.y1, fix.sel(time=-time), transform=crs.PlateCarree())
	ax2 = fig.add_subplot(gs[0,1], projection=crs.NorthPolarStereo(central_longitude=360-45))
	ax2.pcolormesh(var.x1, var.y1, var.sel(time=-time), transform=crs.PlateCarree())

	ax3 = fig.add_subplot(gs[0,2], projection=crs.NorthPolarStereo(central_longitude=360-45))
	ax3.pcolormesh(iso.lon, iso.lat, iso.isochrone, transform=crs.PlateCarree())


	fig.suptitle(f"Time = {time}")
	plt.show()
plot(200)
# d = getIsochrones(dataPath)
# print(d.isochrone.values.tolist())
# plt.contourf(d.isochrone.values[10:,10:])
# plt.show()