# -*- coding: utf-8 -*-
"""
Plot results module.
"""

from tqdm import tqdm
import cartopy.crs as ccrs
import xarray as xr
import numpy as np
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from src.PlotFeatures import *
from src.XMLReader import *
from .Utils import *
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import copy


class Plot:
    """ Parent class to plot a dataArray. """

    def __init__(self):
        self.fig = []
        self.axarr = []
        self.cbar = []
        self.background = []
        self.colorbar = []
        self.extent = []
        self.time_key = []
        self.setup_plot = {}
        self.plot_type = []
        self.n_plot_max_threshold = 12
        self.plot_every_n_step = 4
        self.proj = ccrs.PlateCarree()
        self.cbar_key = []
    
    def setBackground(self, background):
        self.background = background
    
    def setColobar(self, colorbar):
        self.colorbar = colorbar

    def setTimeKey(self, dataArray):
        """ set the time key from dataArray if exist"""
        if hastime(dataArray):
            self.time_key = get_time_key(dataArray)
        else:
            self.time_key = None

    def setSliceTimeDataArray(self, dataArray):
        """ Slice to dataArray to fit into a graph. """

        if hastime(dataArray):
            # If time_size is higher than 12 (for example, 30 days, it plots
            # one of each four days.
            time_size = dataArray.shape[0]
            if time_size > self.n_plot_max_threshold: # Timesteps > 12 slices 
                time_slice = slice(0, -1, int(time_size/self.plot_every_n_step))
                dataArray = dataArray.isel({self.time_key: time_slice})
        return dataArray

    def setFigureAxisLayout(self, dataArray):
        """
        Creates the figure layout to plot.

        Returns:
            None.

        """
        time_plot_flag = hastime(dataArray)
        time_size = dataArray.shape[0]
        if time_plot_flag:
            if time_size < 4:
                nrows = 1
                ncols = time_size
            elif time_size == 4:
                nrows = 2
                ncols = 2
            elif time_size == 6:
                nrows = 2
                ncols = 3
            elif time_size == 8:
                nrows = 2
                ncols = 4
            elif time_size == 12:
                nrows = 3
                ncols = 4
            else:
                nrows = 2
                ncols = 2
        else:
            nrows = ncols = 1

        if nrows == ncols:
            figsize = (17, 15)
        elif ncols > nrows:
            figsize = (20, 15)

        self.fig, self.axarr = plt.subplots(nrows=nrows, ncols=ncols,
                                            figsize=figsize,
                                            subplot_kw={'projection': self.proj})

        if not isinstance(self.axarr, np.ndarray):
            self.axarr = np.array([self.axarr])

    def getScaleBar(self, ax, dataArray):
        """Set the scale bar in km for the current axis"""
        scale_bar_lenght = get_horizontal_scale(dataArray)
        scale_bar(ax, ccrs.PlateCarree(), scale_bar_lenght)


class PlotPolygon(Plot):
    """ Module to plot polygon dataArrays"""

    def __init__(self, polygon_file):
        Plot.__init__(self)
        self.polygon_file = polygon_file
        self.rawGeoDataFrame = gpd.read_file(self.polygon_file).to_crs({"init": 'EPSG:4326'})
        self.rawGeoDataFrame['index'] = np.arange(0, self.rawGeoDataFrame.shape[0])

    def getSetupDict(self, ax):
        
        setup_plot = {
              'cmap': self.colorbar.cmap_key,
              'ax': ax,
              'vmin': self.colorbar.vmin,
              'vmax': self.colorbar.vmax,
              'zorder': 1}

        return setup_plot

    def getPlots(self, dataArray, title, output_filename):
        
        self.setTimeKey(dataArray)
        self.setFigureAxisLayout(dataArray)
        dataArray = self.setSliceTimeDataArray(dataArray)
        self.axarr = self.background.addBackgroundToAxis(self.axarr)
        self.colorbar.addColorbarToFigure(self.axarr, self.fig)
        geoDataFrame = copy.deepcopy(self.rawGeoDataFrame)
        
        time_step = 0
        for ax in self.axarr.flat:

            if hastime(dataArray):
                dataArray_step = dataArray.isel({self.time_key: time_step})
            else:
                dataArray_step = dataArray

            varName = dataArray.name
            geoDataFrame[varName] = dataArray_step.values
            setupPlotDict = self.getSetupDict(ax)

            geoDataFrame.plot(column=varName, **setupPlotDict)
            time_step += 1

        # Creating the suptitle from the filename
        self.fig.suptitle(title, fontsize='x-large')
        # Save the image
        self.fig.savefig(output_filename, dpi=150)
        # Save the shapefile
        shapefile = output_filename.split('.')[0]+'.shp'
        geoDataFrame.to_file(shapefile)
        plt.close()


class PlotGrid(Plot):
    """ Module to plot grid dataArrays"""
    def __init__(self, plot_type):
        Plot.__init__(self)
        self.plot_type = plot_type

    def getSetupDict(self, ax):
        if self.plot_type == 'hist':
            setup_plot = {'ax': ax, 'zorder': 1}

        else:
            setup_plot = {'x':'longitude',
                          'y':'latitude',
                          'ax': ax,
                          'cmap': self.colorbar.cmap_key,
                          'vmin': self.colorbar.vmin,
                          'vmax': self.colorbar.vmax,
                          'zorder': 1,
                          'add_colorbar': False}
        return setup_plot

    def getPlots(self, dataArray, title, output_filename):
        self.setTimeKey(dataArray)
        self.setFigureAxisLayout(dataArray)
        dataArray = self.setSliceTimeDataArray(dataArray)
        self.colorbar.addColorbarToFigure(self.axarr,self.fig)
        self.axarr = self.background.addBackgroundToAxis(self.axarr)
        time_step = 0
        for ax in self.axarr.flat:

            if hastime(dataArray):
                dataArray_step = dataArray.isel({self.time_key: time_step})
            else:
                dataArray_step = dataArray

            setup_plot = self.getSetupDict(ax)
            _ = getattr(dataArray_step.plot, self.plot_type)(**setup_plot)

            self.getScaleBar(ax, dataArray)
            time_step += 1

        # Creating the title from the filename
        self.fig.suptitle(title, fontsize='x-large')
        # fig.tight_layout()
        # Save the title
        self.fig.savefig(output_filename, dpi=150)
        plt.close()



def plotResultsFromRecipe(outDir, xml_recipe):
    """


    Args:
        outDir (TYPE): DESCRIPTION.
        xml_recipe (TYPE): DESCRIPTION.

    Returns:
        None.

    """
    # Read the main xml attributes.
    plot_type = getPlotTypeFromRecipe(xml_recipe)
    polygon_file = getPolygonFileFromRecipe(xml_recipe)
    
    # if there is no plot_type/polygon in the xml, leave the plot module.
    if not (polygon_file or plot_type):
        return       
    
    group_freq, group_type = getPlotTimeFromRecipe(xml_recipe)
    groups = getGroupFromRecipe(xml_recipe)
    methods = getPlotMeasuresFromRecipe(xml_recipe)
    weight_file = getPlotWeightFromRecipe(xml_recipe)
    normalize_method = getNormalizeFromRecipe(xml_recipe)
    

    print('-> Plotting results:')
    print('-> Grouping time steps:', group_freq, group_type)
    print('-> Methods:', methods)
    print('-> Plot_type:', plot_type)
    print('-> Normalizing:', normalize_method)

    # Read the dataset and the variables keys
    ds_raw = xr.open_mfdataset(outDir + '*.nc')
    
    # Split the dataset into groups from variables names. 
    # Process each group together
    ds_list = get_variable_groups_from_dataset(ds_raw)
    
    # We prepare the plotter instance with the backgrounds, axis and 
    # all the stuff to work properly.
    if polygon_file:
        plotter = PlotPolygon(polygon_file[0])
        extent = get_extent(plotter.rawGeoDataFrame)
    elif plot_type:
        plotter = PlotGrid(plot_type[0])
        extent = get_extent(ds_raw)

    # Initialize the background images for all the plots based on the extents.
    # This avoid the heavy resources consumption of download detailed backgrounds .
    bkg_images = Background(extent)
    bkg_images.initialize()
    plotter.setBackground(bkg_images)
    
    for ds in ds_list:
        
        variables = list(ds.keys())
        # Select just the desire group of variables. For example, choose 
        # a type of particle or whatever you want to group all your variables.
        if groups:
            variables = groupByWords(variables, groups)
            if len(variables) == 0:
                print('-> You did an empty group. Stopping')
                raise ValueError
            ds = ds[variables]
    
        # Normalization of the dataset.
        if normalize_method:
            normalizer = Normalizer(normalize_method, ds)
            normalizer.setFactor()
            normalizer.setMethodName()
            ds = normalizer.getNormalizedDataset(ds)
        
        # colorbar with common values for all the plots
        # to compare them correctly.
        cb = Colorbar()
        cb.setMinMaxColor(ds)
        cb.setCmapKey()
        plotter.setColobar(cb)
        
        for idxvar, variable in enumerate(tqdm(variables)):
            da = ds[variable].load()
            if normalize_method:
                units = normalizer.getNormalizedUnits(da.units)
            else:
                units = da.units
            
            # add the units to the colorbar
            cb.setUnits(units)
    
            if weight_file:
                if idxvar == 0:
                    print("-> Weighting sources with:", weight_file[0])
                    print("   %-40s | %9s" % ('Source', 'Weight'))
                da = weight_dataarray_with_csv(da, weight_file[0])
    
            if 'depth' in da.dims:
                da = da.isel(depth=-1)
    
            methods_list = []
            for method in methods:
                if group_freq != 'all':
                    if isgroupable(da, group_type, group_freq, method):
                        da = group_resample(da, group_type, group_freq)
    
                da = getattr(da, method)(dim='time')
                methods_list.append(method)
    
            da = da.where(da != 0)  # Values with 0 consider as missing value.
    
            output_filename = outDir + '-'.join(methods_list) + '-' + variable + '.png'
            title = get_title_methods(methods_list, variable)
    
            if polygon_file:
                plotter.getPlots(da, title, output_filename)
            else:
                plotter.getPlots(da, title, output_filename)
