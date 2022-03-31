import os
import numpy as np
from numpy.lib.function_base import diff
from numpy.lib.twodim_base import mask_indices
import pandas as pd
from osgeo import gdal
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.lines import Line2D
import geopandas as gpd
from scipy.sparse import dok
from sklearn import metrics
import math
import glob

WD = "D:/projects/AgoraNatura/agora_natura"
os.chdir(WD)

## Raster mask to identify areas per landkreis
LDK_MSK = f"{WD}/result_analysis/data/administrative_masks/landkreise_1000.asc"

## Shapefile for plotting
LDK_SHP = f"{WD}/result_analysis/data/vector/GER_landkreise_25832.shp"

## Path for general crop area mask
# CROP_MSK_PTH2 = "/beegfs/jaenicke/klimertrag/result_analysis/data/crop_masks/CTM_17-19_mask_1000m_25832.asc"
## Path for crop specific masks
CROP_MSK_PTH2 = f"{WD}/result_analysis/data/crop_masks/CM_2017-2019_{0}_1000m_25832_q3.asc"

## Reference tables with observed yields for whole of Germany and for each district
OBS_PTH_LDK = f"{WD}/result_analysis/data/reference_tables/yields/Ertraege_Landkreise_10_Frucharten_1999-2020_detrended.csv"
OBS_PTH_GER = f"{WD}/result_analysis/data/reference_tables/yields/Ertraege_Deutschland_10_Frucharten_1999-2020_detrended.csv"
## Column of reference table that should be used 'yield_obs_detr'/'yield_obs' (detrended values vs original values)
YIELD_OBS_COL = 'yield_obs_detr'

YIELD_PTH = "/beegfs/jaenicke/klimertrag_temp/raster/sim-yields/11_final_results"
YIELD_PTH = ""

## Paths where the simulation results are stored. Is not in repository because they were too big and changed too often.
OUT_FOLDER_TABLES = "D:/projects/KlimErtrag/tables/11_final_results"
OUT_FOLDER_FIGURES = "D:/projects/KlimErtrag/figures/klimertrag_landkreise/11_final_results/yield_validation"

## Run IDs of MONICA simulation runs
RUN_IDS = [18, 54, 165, 90, 167]

## Dictionary with crop abbreviation and name for each run
RUN_IDS = {
    54: {'crop':'WW',
         'crop_name': 'Winter wheat'},
    18: {'crop':'WB',
         'crop_name': 'Winter barley'},
    90: {'crop':'WR',
         'crop_name': 'Oilseed rape'},
    167: {'crop':'SM',
         'crop_name': 'Silage maize'},
    165: {'crop':'WRye',
         'crop_name': 'Winter rye'}
}

## Dictionary with year list for plottng and
## fresh matter conversion factor (conversion factor is a relict of an older version, not needed anymore)
## [Range of years to validation, Conversion factor fresh matter to dry matter, absolute max error for plotting]
CROPS = {
    'WW': [list(range(1999, 2020)), 86],
    'SM': [list(range(1999, 2020)), 32],
    'WB': [list(range(1999, 2020)), 86, 2500],
    'WR': [list(range(1999, 2020)), 91],
    'WRye': [list(range(1999, 2020)), 86],
    'PO': [list(range(1999, 2020)), 22.5],
    'SB': [list(range(1999, 2020)), 86],
    'SU': [list(range(1999, 2020)), 22.5]
}


def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i / inch for i in tupl[0])
    else:
        return tuple(i / inch for i in tupl)


def d_modified(cal, obs, order=1):
    import warnings
    """
    Modify Index of Agreement (Willmott et al., 1984) range from 0.0 to 1.0
    and the closer to 1 the better the performance of the model.
    :param cal: (N,) array_like of calculated values
    :param obs: (N,) array_like of observed values
    :param order: exponent to be used in the computation. Default is 1
    :return: Modified Index of Agreement between 'cal' and 'obs'
    .. math::
        r = 1 - \\frac{\\sum (O_i - C_i)^n}
                      {\\sum (\\vert C_i - m_O \\vert + \\vert O_i - m_O \\vert)^n}
    where :math:`m_O` is the mean of the vector :math:`O` of observed values.

    Found @ https://github.com/SimonDelmas/goodness_of_fit/blob/master/goodness_of_fit/__init__.py
    """

    obs_mean = np.nanmean(obs)
    denominator = np.nansum(
        np.power(np.abs(cal - obs_mean) + np.abs(obs - obs_mean), order)
    )
    if denominator == 0.0:
        warnings.warn("Index of agreement potential error is null! Return NaN.")
        return np.nan

    nominator = np.nansum(np.abs(np.power(obs - cal, order)))
    return 1 - (nominator / denominator)


def calc_perf_meas(df):
    print('')

    ##lst = [["MAE", "nMAE", "RMSE", "MBE", "nMBE", "d_modified", "R_squared"]]

    pred = 'yield_sim'
    ref = YIELD_OBS_COL  # 'yield_obs_detr'
    df = df.dropna(subset=[pred, ref])
    pred_data = df[pred].to_numpy()
    ref_data = df[ref].to_numpy()

    if len(pred_data) > 0:
        mae = round(metrics.mean_absolute_error(ref_data, pred_data), 3)
        nmae = round(metrics.mean_absolute_error(ref_data, pred_data) / np.mean(ref_data), 3)
        rmse = round(math.sqrt(metrics.mean_squared_error(ref_data, pred_data)), 3)
        mbe = np.mean(pred_data - ref_data)
        nmbe = np.mean(pred_data - ref_data) / np.mean(ref_data)
        agr_ind = round(d_modified(ref_data, pred_data), 3)

        corr_matrix = np.corrcoef(np.array(ref_data), np.array(pred_data))
        corr = corr_matrix[0, 1]
        rsq = round(corr ** 2, 2)

        out_lst = [mae, nmae, rmse, mbe, nmbe, agr_ind, rsq]

        return out_lst

    else:
        print('No predicted values in df.')


def plot_yield_difference_in_grid(run_ids, diff_col, title, out_pth, max_val=None):
    """
    Plots the difference maps between observed and simulated yield values in a grid.
    """

    ## Some plot characteristics
    font_size = 12
    plt.rcParams['legend.title_fontsize'] = f'{font_size}'
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams["font.family"] = "DejaVu Sans"

    ## Initiate plot
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(cm2inch(16, 16)),
                            gridspec_kw=dict(height_ratios=[8, 8]))

    ## Go over run_ids and plot
    for r, run_id in enumerate(run_ids):

        crop = run_ids[run_id]["crop"]
        crop_name = run_ids[run_id]["crop_name"]

        ix = np.unravel_index(r, axs.shape)

        ## Open simulation results
        df = pd.read_csv(f"{OUT_FOLDER_TABLES}/{crop}_nuts3_GER_sim-obs_yields_years_aggr_{run_id}.csv")

        ## Open shapefile
        shp = gpd.read_file(LDK_SHP)
        shp.cca_2 = shp.cca_2.apply(int)

        ## ToDo: Move this into calculation of tables
        ## Calculate the difference between the observed and the simulated values
        df['diff_obsim'] = round((df[YIELD_OBS_COL] - df['yield_sim']) / df[YIELD_OBS_COL] * 100, 1)

        ## Only use data from current crop
        df_sub = df.loc[(df['Crop'] == crop)]

        ## Merge yields with shp
        shp_plt = pd.merge(shp, df_sub, how='left', left_on='cca_2', right_on='ID')

        ## Remove Oberallgäu
        shp_plt.loc[shp_plt['cca_2'] == 9780, 'diff_obsim'] = None
        shp_plt.loc[shp_plt['cca_2'] == 9780, YIELD_OBS_COL] = None
        shp_plt.loc[shp_plt['cca_2'] == 9780, 'yield_sim'] = None

        if not max_val:
            max_val = max(shp_plt[diff_col].max(), abs(shp_plt[diff_col].min()))
        else:
            max_val = max_val

        v_min = -35
        v_max = 30

        nbounds = ((-v_min + v_max)/5) + 1

        cmap = plt.cm.RdYlBu
        cmaplist = [cmap(i) for i in range(cmap.N)]
        cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, cmap.N)
        # define the bins and normalize
        bounds = np.linspace(v_min, v_max, int(nbounds))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        # shp.plot(column=diff_col, ax=axs, legend=True)
        shp_plt.plot(column=diff_col, ax=axs[ix], vmin=v_min, vmax=v_max, cmap=cmap)  #legend=True, cax=cax,
                 # legend_kwds={'label': "Ertrag Trockenm [kg/ha]"})

        shp_plt.plot(edgecolor='black', facecolor="none", ax=axs[ix], lw=0.1)
        axs[ix].set_title(crop_name)
        axs[ix].set_axis_off()

        max_val = None

    ix = np.unravel_index(r+1, axs.shape)
    axs[ix].set_axis_off()

    ## Add colorbar
    ax4 = fig.add_axes([0.68, 0.1, 0.01, 0.31])  #[left, bottom, width, height]
    # ax4.annotate(text="Obs.-Sim.\n    [%]", xy=(0.89, 0.89), xycoords='figure fraction', fontsize=10)

    cb = mpl.colorbar.ColorbarBase(ax4, cmap=cmap, norm=norm, spacing='proportional',
                                   orientation='vertical', label='Observation - Simulation [%]')
    cb.set_ticks(bounds)
    cb.ax.yaxis.set_tick_params(color="white")
    cb.set_ticklabels(["<-35", "-30","-25", "-20", "-15", "-10", "-5", "0", "5", "10", "15", "20", "25", ">30"])
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)

    plt.tight_layout()
    plt.show()
    fig.savefig(out_pth)
    plt.close()

def plot_legend(v_min, v_max, width, height, label, out_pth):

    fig, ax = plt.subplots(figsize=(cm2inch(width, height)))

    nbounds = int(((-v_min + v_max) / 5) + 1)

    cmap = plt.cm.RdYlBu  # RdYlGn
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'Custom cmap', cmaplist, cmap.N)
    # define the bins and normalize
    bounds = np.linspace(v_min, v_max, nbounds)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax,
                      orientation='horizontal', fraction=0.5, label=label)
    cb.set_ticks(bounds)
    cb.ax.yaxis.set_tick_params(color="white")
    cb.set_ticklabels(["<-35", "-30", "-25", "-20", "-15", "-10", "-5", "0", "5", "10", "15", "20", "25", ">30"])
    cb.ax.tick_params(labelsize=10)
    cb.outline.set_visible(False)

    plt.tight_layout()

    plt.savefig(out_pth, dpi=600, bbox_inches='tight')
    plt.close()


def plot_obs_vs_sim_in_grid(run_ids, obs_col, sim_col, out_pth):
    font_size = 10
    plt.rcParams['legend.title_fontsize'] = f'{font_size}'
    plt.rcParams['legend.fontsize'] = 9
    plt.rcParams["font.family"] = "DejaVu Sans"

    ncols = math.ceil(len(run_ids) / 2)
    nrows = math.ceil(len(run_ids) / 2)

    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=cm2inch(16.0, 22))

    for r, run_id in enumerate(run_ids):
        crop = run_ids[run_id]["crop"]
        crop_name = run_ids[run_id]["crop_name"]

        ix = np.unravel_index(r, axs.shape)
        df = pd.read_csv(f"{OUT_FOLDER_TABLES}/{crop}_nuts3_GER_sim-obs_yields_years_aggr_{run_id}.csv")

        df = df[df[obs_col].notna()]
        df = df[df[sim_col].notna()]

        df['diff_obsim'] = df[YIELD_OBS_COL] - df['yield_sim']
        df = df.loc[(df['Crop'] == crop)]

        ## Calculate performance measures
        agr_ind = round(d_modified(df[obs_col], df[sim_col]), 2)
        rmse = int(math.sqrt(metrics.mean_squared_error(df[obs_col], df[sim_col])))
        nmbe = round(np.mean(np.array(df[obs_col]) - np.array(df[sim_col])) / np.mean(np.array(df[obs_col])), 2)
        nmae = round(metrics.mean_absolute_error(np.array(df[obs_col]), np.array(df[sim_col])) / np.mean(np.array(df[obs_col])), 2)
        corr_matrix = np.corrcoef(np.array(df[obs_col]), np.array(df[sim_col]))
        corr = corr_matrix[0, 1]
        rsq = round(corr ** 2, 2)

        axs[ix].scatter(x=df[sim_col], y=df[obs_col], s=3)
        axs[ix].annotate(text=f"RMSE: {rmse}", xy=(0.051, 0.6), xycoords='axes fraction')
        axs[ix].annotate(text=f"nMAE: {nmae}", xy=(0.051, 0.67), xycoords='axes fraction')
        axs[ix].annotate(text=f"nMBE: {nmbe}", xy=(0.051, 0.74), xycoords='axes fraction')
        axs[ix].annotate(text=f"R²: {rsq}", xy=(0.051, 0.81), xycoords='axes fraction')
        axs[ix].annotate(text=f"d: {agr_ind}", xy=(0.051, 0.9), xycoords='axes fraction')

        max_val = max(df[sim_col].max(), df[obs_col].max())
        min_val = min(df[sim_col].min(), df[obs_col].min())

        axs[ix].plot([min_val, max_val], [min_val, max_val], linewidth=0.5, c='black', linestyle='--')

        axs[ix].set_title(crop_name, loc='left', fontsize=font_size)
        # ax.set_xlabel('Reference')
        # ax.set_ylabel('Prediction')
        axs[ix].set_xlabel('Simulation')
        axs[ix].set_ylabel('Observation')

    ix = np.unravel_index(r + 1, axs.shape)
    axs[ix].set_axis_off()

    plt.tight_layout()
    plt.show()
    fig.savefig(out_pth)
    plt.close()


def plot_observat_simulation_time_series_in_grid(run_ids, obs_col, sim_col, out_pth):
    font_size = 10
    plt.rcParams['legend.title_fontsize'] = f'{font_size}'
    plt.rcParams['legend.fontsize'] = 8
    plt.rcParams["font.family"] = "DejaVu Sans"

    ncols = math.ceil(len(run_ids) / 2)
    nrows = math.ceil(len(run_ids) / 2)

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=cm2inch(16.0, 22))

    for r, run_id in enumerate(run_ids):
        crop = run_ids[run_id]["crop"]
        crop_name = run_ids[run_id]["crop_name"]

        ix = np.unravel_index(r, axs.shape)
        df_ts = pd.read_csv(f"{OUT_FOLDER_TABLES}/{crop}_GER_sim-obs_yields_years_ts_{run_id}.csv")
        min_year = 1990
        max_year = 2020

        df = df_ts.loc[(df_ts['Crop'] == crop)]
        df = df.loc[(df['Year'] >= min_year) & df['Year'] <= max_year].copy()
        df = df[df[obs_col].notna()]
        df = df[df[sim_col].notna()]

        max_val = max([df[obs_col].max(), df[sim_col].max()])
        min_val = min([df[obs_col].min(), df[sim_col].min()])

        ymax = max_val + 0.1 * max_val
        ymin = min_val - 0.1 * min_val

        ## Calculate performance measures
        agr_ind = round(d_modified(df[obs_col], df[sim_col]), 2)
        rmse = int(math.sqrt(metrics.mean_squared_error(df[obs_col], df[sim_col])))
        nmbe = round(np.mean(np.array(df[obs_col]) - np.array(df[sim_col])) / np.mean(np.array(df[obs_col])), 2)
        nmae = round(
            metrics.mean_absolute_error(np.array(df[obs_col]), np.array(df[sim_col])) / np.mean(np.array(df[obs_col])),
            2)
        corr_matrix = np.corrcoef(np.array(df[obs_col]), np.array(df[sim_col]))
        corr = corr_matrix[0, 1]
        rsq = round(corr ** 2, 2)

        obs = np.array(df[obs_col])
        sim = np.array(df[sim_col])
        differences = sim - obs
        sum_diff = np.sum(sim - obs)
        sum_obs = np.sum(obs)
        nmbe = round(sum_diff/sum_obs, 2)

        corr_matrix = np.corrcoef(np.array(df[obs_col]), np.array(df[sim_col]))
        corr = corr_matrix[0, 1]
        rsq = round(corr ** 2, 2)

        axs[ix].plot(df['Year'], df[obs_col], c='red')
        axs[ix].plot(df['Year'], df[sim_col], c='blue')
        axs[ix].set_ylim([ymin, ymax])


        text = f"RMSE:{rmse}, nMAE:{nmae},\nR²:{rsq}, nMAE:{nmae}, nMBE:{nmbe}"
        axs[ix].annotate(text=text, xy=(0.021, 0.027), xycoords='axes fraction', size=9)
        axs[ix].set_title(crop_name, loc='left', fontsize=font_size)

    ix = np.unravel_index(r + 1, axs.shape)
    axs[ix].set_axis_off()

    custom_lines = [Line2D([0], [0], color='red', lw=1), Line2D([0], [0], color='blue', lw=1)]
    legend_labels = ['Observation', 'Simulation']
    axs[(0, 0)].legend(custom_lines, legend_labels, loc='upper left')



    # for ix in [(0,0),(1,0),(2,0),(3,0)]:
    #     axs[ix].set_ylabel('Ertrag [kg/ha]')
    #
    # for ix in [(3,0),(3,1)]:
    #     axs[ix].set_xlabel('Jahr')

    fig.tight_layout()
    plt.show()
    fig.savefig(out_pth)
    plt.close()


def plot_observat_simulation_time_series(df, obs_col, sim_col, min_year, max_year, title, out_pth):
    font_size = 12
    plt.rcParams['legend.title_fontsize'] = f'{font_size}'
    plt.rcParams["font.family"] = "DejaVu Sans"

    df = df.loc[(df['Year'] >= min_year) & df['Year'] <= max_year].copy()

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=cm2inch(16, 8))

    ax.plot(df['Year'], df[obs_col], c='red')
    ax.plot(df['Year'], df[sim_col], c='blue')

    ax.set_title(title, loc='left', fontsize=font_size)
    # ax.set_xlabel('Year')
    # ax.set_ylabel('Yield dry matter [kg/ha]')
    ax.set_xlabel('Jahr')
    ax.set_ylabel('Ertrag Trockenmasse [kg/ha]')

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.01,
                     box.width, box.height * 0.95])

    custom_lines = [Line2D([0], [0], color='red', lw=1), Line2D([0], [0], color='blue', lw=1)]
    # legend_labels = ['Reference', 'Simulation']
    legend_labels = ['Referenz', 'Simulation']
    ax.legend(custom_lines, legend_labels, loc='upper center', bbox_to_anchor=(0.15, -0.08),
              ncol=2)

    plt.tight_layout()
    # plt.show()
    plt.savefig(out_pth)
    plt.close()


def main():
    # plot_yield_difference_in_grid(run_ids=RUN_IDS,
    #                               diff_col='diff_obsim',
    #                               title='All crops',
    #                               out_pth=r"D:\projects\AgoraNatura\figures\yield_difference_maps.png", max_val=None)

    out_pth = fr"D:\projects\AgoraNatura\figures\scatter-spatial_obs-sim.png"
    plot_obs_vs_sim_in_grid(run_ids=RUN_IDS,
                            obs_col=YIELD_OBS_COL,
                            sim_col='yield_sim',
                            out_pth=out_pth)

    out_pth = fr"D:\projects\AgoraNatura\figures\timeseries_obs-sim.png"
    plot_observat_simulation_time_series_in_grid(run_ids=RUN_IDS,
                            obs_col=YIELD_OBS_COL,
                            sim_col='yield_sim_geravg',
                            out_pth=out_pth)

    # plot_legend(v_min=-35,
    #             v_max=30,
    #             width=13,
    #             height=0.3,
    #             label="Observationen-Simulationen [%]",
    #             out_pth=rf"D:\projects\KlimErtrag\figures\klimertrag_landkreise\11_final_results\yield_validation\legend_obs-sim_-25_to_+30.jpg")

if __name__ == '__main__':
    main()
