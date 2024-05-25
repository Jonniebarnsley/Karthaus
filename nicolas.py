import xarray as xr
import matplotlib.pyplot as plt


dataPath = "../Data/"
for i in ["very_low", "low", "mid", "high"]:
	try:
		dd = xr.open_dataset(dataPath + f"Isochrones/9.0-8.5_ka_{i}_confidence_isochrone.nc").isochrone
		if "d" not in locals():
			d = dd
		else:
			d = d + dd
	except:
		print(i, "fail")
	

plt.contourf(d)
plt.show()