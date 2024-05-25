import xarray as xr
import matplotlib.pyplot as plt

dataPath = "../Data/"

def getIsochrones(dataPath, name):
	for i, confidence in enumerate(["very_low", "low", "mid", "high"]):
		try:
			dd = xr.open_dataset(dataPath + f"Isochrones/{name}_ka_{confidence}_confidence_isochrone.nc").isochrone
			if "d" not in locals():
				d = dd
			else:
				d = d + dd*(i+1)
		except:
			print(f"Isochrones {name} does not have confidence {confidence}")
	return d

d = getIsochrones(dataPath, "9.0-8.5")
plt.contourf(d)
plt.show()