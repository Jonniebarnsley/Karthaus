import xarray as xr
import matplotlib.pyplot as plt

dataPath = "../Data/"

def getIsochrones(dataPath):
	d = xr.open_dataset(dataPath + "isochrone_buffers/PaleoGrIS_1.0_isochrone_buffers_2km_age.nc")
	return d

def getMF(dataPath):
	return xr.open_dataset(dataPath + "GISModel/fixed-seasonality.nc")

def getMV(dataPath):
	return xr.open_dataset(dataPath + "GISModel/variable-seasonality.nc")


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


def plot():


d = getIsochrones(dataPath)
print(d)
plt.contourf(d.isochrone)
plt.show()