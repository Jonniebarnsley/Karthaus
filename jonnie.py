import xarray as xr
import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
from scipy import ndimage
from pathlib import Path

def get_contour(model: xr.Dataset, year: int) -> xr.DataArray:

    # load file into memory and select time
    timeslice: xr.Dataset = model.sel(time=-year) # negative year since time axis is from -22k to 0
    
    # create mask of ice sheet height > 0
    height: xr.DataArray = timeslice.h
    mask: xr.DataArray = xr.where(height > 0, 1, 0)

    # use binary erosion to get trace of ice sheet edge and set all else to NaN
    contour: xr.DataArray = mask - ndimage.binary_erosion(mask, iterations=3)
    contour_no_zero: xr.DataArray = contour.where(contour != 0)

    # create DataArray for output
    lon: np.array = model.lon
    lat: np.array = model.lat
    da: xr.DataArray = xr.DataArray(data = contour_no_zero, coords = {'lon': lon, 'lat': lat})

    return da

def get_iso_extent(iso: xr.Dataset, time: int) -> xr.DataArray:

    # create mask of coordinates with isochrone date less than specified time
    isochrone: xr.DataArray = iso.isochrone
    mask: xr.DataArray = xr.where(isochrone <= time, 1, np.nan)
    
    # create DataArray for output
    da: xr.DataArray = xr.DataArray(mask, coords={'lon':iso.lon, 'lat':iso.lat})

    return da

def plot_model(ax: mpl.pyplot.axis, model_path: str | Path, time: int, color: str='red') -> None:

    # read model data and get contour
    model: xr.Dataset = xr.open_dataset(model_path)
    mask: xr.DataArray = get_contour(model, time)

    # set colour
    cmap: mpl.colors.Colormap = mpl.colors.ListedColormap([color])

    # plot on axis
    ax.pcolormesh(model.lon, model.lat, mask, transform=ccrs.PlateCarree(), cmap=cmap)

def plot_iso(ax: mpl.pyplot.axis, iso_path: str | Path, time: int, color: str = 'lightgrey') -> None:

    # read iso data
    iso: xr.Dataset = xr.open_dataset(iso_path)
    mask: xr.DataArray = get_iso_extent(iso, time)

    # get lat/lon for plotting
    lon = iso.lon
    lat = iso.lat 

    # set colour
    cmap = mpl.colors.ListedColormap([color])

    # plot on axis
    ax.pcolormesh(lon, lat, mask, transform=ccrs.PlateCarree(), cmap=cmap)