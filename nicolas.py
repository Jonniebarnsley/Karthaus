import xarray as xr
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import cartopy.crs as crs

dataPath = "../Data/"
Rt = 6371e3

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

# def getArea():
# 	data = getIsochrones(dataPath)
# 	print(data)
# 	from geopy.distance import geodesic
# 	def haversine_distance(coord1, coord2):
# 		"""Calculate the great-circle distance between two points on the Earth surface."""
# 		return geodesic(coord1, coord2).meters

# 	def calculate_grid_area(lat, lon):
# 		"""Calculate the area of each grid cell in square meters."""
# 		nlat, nlon = lat.shape
# 		area = np.zeros((nlat, nlon))
# 		for i in range(nlat - 1):
# 			print(i)
# 			for j in range(nlon - 1):
# 				p1 = (lat[i, j], lon[i, j])
# 				p2 = (lat[i + 1, j], lon[i + 1])
# 				p3 = (lat[i, j + 1], lon[i, j + 1])
				
# 				# Calculate distances between the points
# 				dy = haversine_distance(p1, p2)
# 				dx = haversine_distance(p1, p3)
				
# 				# Assume the grid cell is approximately rectangular
# 				area[i, j] = dy * dx

# 	grid_area = calculate_grid_area(data['lat'].values, data['lon'].values)

# 	data['area'] = (('lat', 'lon'), grid_area)
# 	# ds["S"] = (('lat', "lon"), Rt**2*np.cos(np.meshgrid(ds.lon, ds.lat)[1]*np.pi/180)*(0.01*np.pi/180)**2)

# getArea()
# exit()

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

def get_contour(model_filepath, year):
    # load file into memory and select time
    file = xr.open_dataset(model_filepath)
    timeslice = file.sel(time=-year) #negative year since time axis is from -22k to 0
    
    # create mask of ice sheet height > 0
    height = timeslice.h
    mask = xr.where(height > 0, 1, 0)

    # use binary erosion to get trace of ice sheet edge
    contour = mask - ndimage.binary_erosion(mask)

    # save as xarray DataArray
    x = mask.x1
    y = mask.y1
    da = xr.DataArray(data = contour, coords = {'x': x, 'y': y})

    return da

def dist(lat1, lon1, lat2, lon2):
	dlat = np.pi/180*(lat2-lat1)/2
	dlon = np.pi/180*(lon2-lon1)/2
	mlat = np.pi/180*(lat1+lat2)/2
	dst = 2*Rt*np.arcsin( (np.sin(dlat)**2 + (1 - np.sin(dlat)**2 - np.sin(mlat)**2)*np.sin(dlon)**2)**0.5 )
	return dst

# def getListClosestPoint(iso, indx, indy, maxdist):
# 	lat1 = iso.lat[indx, indy]
# 	lon1 = iso.lon[indx, indy]
# 	L = [[lat1, lon1]]
# 	indices = np.where(iso.isochrone)
# 	indices = list(zip(indices[0], indices[1]))
# 	for k in indices:
# 		i,j = k
# 		lat2 = iso.lat[i, j]
# 		lon2 = iso.lon[i, j]
# 		if dist(lat1, lon1, lat2, lon2) <= maxdist:
# 			L.append([lat2, lon2])
# 	return np.array(L)

# def getNormalFromPoints(Points):
# 	lat = Points[0,:]
# 	lon = Points[1,:]

# 	mlat = np.mean(lat)
# 	mlon = np.mean(lon)

# 	slope, intercept = np.polyfit(lat, lon, 1)
# 	return (mlat, mlon), (-slope, 1)

# def getDistance2Model(model, point, normal):
# 	indices = np.where(model)
# 	indices = list(zip(indices[0], indices[1]))
# 	minDistance = 1e99
# 	for k in indices:
# 		i,j = k
# 		latPoint = model.x1[i, j]
# 		lonPoint = model.y1[i, j]

# 		a = normal[1]
# 		b = -normal[0]
# 		c = normal[0]*point[1] - normal[1]*point[0]


# 		# distance =  np.abs( a* ) / np.sqrt( a*a + b*b )
# 		if distance < minDistance:
# 			minDistance = distance
# 	return minDistance

# def getScore(time=10000):
# 	fix = get_contour(dataPath+"GISModel/fixed-seasonality.nc", 10000)
# 	model = fix
# 	print(fix)
# 	indices = np.where(model)
# 	indices = list(zip(indices[0], indices[1]))
# 	print(indices)
# 	quit()

# 	iso = getIsochrones(dataPath)
# 	iso_ = iso.isochrone.values
# 	iso_[iso_!=time]=0
# 	iso["isochrone"] = (("x","y"), iso_)
# 	print(iso)
# 	# cs = plt.pcolormesh(iso.isochrone)
# 	# plt.colorbar(cs)
# 	# plt.show()

# 	print("getListClosestPoint")
# 	Points = getListClosestPoint(iso, 1160, 990, 10e3)
# 	print("getNormalFromPoints")
# 	point, normal = getNormalFromPoints(Points)
# 	print("getDistance2Model")
# 	getDistance2Model(fix, point, normal)

def getScore(time=10000):
	fix = get_contour(dataPath+"GISModel/fixed-seasonality.nc", 10000)
	fixScoreMap = fix
	fixPoints = np.where(fix)
	fixPoints = list(zip(fixPoints[0], fixPoints[1]))
	# fixPoints = np.array([(float(fix.y[i[0]].values), float(fix.x[i[1]].values)) for i in fixPoints])

	iso = getIsochrones(dataPath)
	iso_ = iso.isochrone.values
	iso_[iso_!=time]=0
	iso["isochrone"] = (("x","y"), iso_)
	print(iso)
	print("\n")	
	isoPoints = np.where(iso.isochrone)
	isoPoints = list(zip(isoPoints[0], isoPoints[1]))
	# isoPoints = np.array([(float(iso.lat[i[0], i[1]].values), float(iso.lon[i[0], i[1]].values)) for i in isoPoints])

	print("\n")
	fixScore = 0
	for i in range(len(fixPoints)):
		lat2 = fix.lat[fixPoints[i][0], fixPoints[i][1]]
		lon2 = fix.lon[fixPoints[i][0], fixPoints[i][1]]
		minDist = 1e99
		for j in range(len(isoPoints)):
			lat1 = fix.y[isoPoints[j][0]]
			lon1 = fix.x[isoPoints[j][1]]
			distToPoint = dist(lat1, lon1, lat2, lon2)
			if distToPoint < minDist:
				minDist = distToPoint

		fixScore += minDist
		fix.score[fixPoints[i][0], fixPoints[i][1]] = minDist
		fix.print(minDist)

	cs = plt.pcolormesh(fix.score)
	plt.colorbar(cs)
	plt.show()






getScore()
exit()

# fix = get_contour(dataPath+"GISModel/fixed-seasonality.nc", 10000)
# print(fix)
# cs = plt.pcolormesh(fix)
# plt.colorbar(cs)
# plt.show()
# exit()



iso = getIsochrones(dataPath)
iso = iso.isochrone.values
iso[iso!=10000.]=0

cs = plt.pcolormesh(iso)
plt.colorbar(cs)
plt.show()


# plot(200)
# d = getIsochrones(dataPath)
# print(d.isochrone.values.tolist())
# plt.contourf(d.isochrone.values[10:,10:])
# plt.show()