import xarray as xr
from scipy import ndimage
from pathlib import Path

def get_contour(model: xr.Dataset, year: int) -> xr.DataArray:

    # load file into memory and select time
    timeslice = model.sel(time=-year) # negative year since time axis is from -22k to 0
    
    # create mask of ice sheet height > 0
    height = timeslice.h
    mask = xr.where(height > 0, 1, 0)

    # use binary erosion to get trace of ice sheet edge and set all else to NaN
    contour = mask - ndimage.binary_erosion(mask, iterations=2)
    contour_no_zero = contour.where(contour != 0)

    # save as xarray DataArray
    lon = model.lon
    lat = model.lat
    da = xr.DataArray(data = contour_no_zero, coords = {'lon': lon, 'lat': lat})

    return da