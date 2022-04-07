import os
import numpy as np
import pandas as pd
from osgeo import gdal
import glob

WD = "D:/projects/AgoraNatura/agora_natura"
os.chdir(WD)

LDK_MSK = "D:/projects/AgoraNatura/agora_natura/result_analysis/data/administrative_masks/landkreise_1000.asc"
LDK_SHP = "D:/projects/AgoraNatura/agora_natura/result_analysis/data/vector/GER_landkreise_25832.shp"
# CROP_MSK_PTH2 = "D:/projects/AgoraNatura/agora_natura/result_analysis/data/crop_masks/CM_2017-2019_{0}_1000m_25832_q3.asc"
CROP_MSK_PTH2 = "D:/projects/AgoraNatura/agora_natura/result_analysis/data/crop_masks/CTM_17-19_mask_1000m_25832.asc"


YIELD_PTH = "D:/projects/AgoraNatura/raster/calibration_results"
RUN_IDS = [20, 21 ] #18, 54, 90, 165, 167]

OBS_PTH_LDK = f"D:/projects/AgoraNatura/agora_natura/result_analysis/data/reference_tables/yields/Ertraege_Landkreise_10_Frucharten_1999-2020_detrended.csv"
OBS_PTH_GER = f"D:/projects/AgoraNatura/agora_natura/result_analysis/data/reference_tables/yields/Ertraege_Deutschland_10_Frucharten_1999-2020_detrended.csv"

OUT_FOLDER_TABLES = "D:/projects/AgoraNatura/tables/final_results"
OUT_FOLDER_FIGURES = "D:/projects/AgoraNatura/figures/final_results"

YIELD_OBS_COL = 'yield_obs_detr'

## [Range of years to validation, Conversion factor fresh matter to dry matter, absolute max error for plotting]
CROPS = {
    'WW': [list(range(1999, 2020)),86],
    'SM': [list(range(1999, 2020)),32],
    'WB': [list(range(1999, 2020)),86, 2500],
    'WR': [list(range(1999, 2020)),91],
    'WRye': [list(range(1999, 2020)),86],
    'PO': [list(range(1999, 2020)),22.5],
    'SB': [list(range(1999, 2020)),86],
    'SU': [list(range(1999, 2020)),22.5],
    'CLALF': [list(range(1999, 2020)),100]
}

def merge_observation_simulation_years_aggregated():
    ## Loop run_ids
    for run_id in RUN_IDS:

        yield_pth = f"{YIELD_PTH}/{run_id}"

        if not os.path.exists(yield_pth):
            print(f"Path to yield simulations with run-ID {run_id} does not exist. \n {yield_pth}. \n")
            return

        ## Get crop name of current run ID
        crop_name_dict = {
            'wheatwinterwheat':'WW',
            'barleywinterbarley':'WB',
            'ryewinterrye':'WRye',
            'maizesilagemaize':'SM',
            'sugarbeet':'SU',
            'potatomoderatelyearlypotato':'PO',
            'rapewinterrape':'WR',
            'barleyspringbarley':'SB',
            'alfalfaclovergrassleymix':'CLALF'
        }
        search_term = rf'{yield_pth}/*Yield*.asc'
        file_lst = glob.glob(search_term)
        if file_lst:
            file_pth = file_lst[0]
            crop_name = os.path.basename(file_pth).split('_')[0]
            crop = crop_name_dict[crop_name]
        else:
            continue
        
        ## Get year list from global variables
        ## And frischmasse-trockenmasse conversion factor
        year_lst = CROPS[crop][0]
        conv_factor = CROPS[crop][1]
        yield_arr_lst = []

        ## Open Landkreise Raster and
        ldk_ras = gdal.Open(LDK_MSK)
        ldk_arr = ldk_ras.ReadAsArray()
        ndv_ldk = ldk_ras.GetRasterBand(1).GetNoDataValue()

        ldk_ids = np.unique(ldk_arr)
        ldk_ids = ldk_ids[ldk_ids != ndv_ldk]
        ldk_ids = ldk_ids[ldk_ids != 0]

        ## List for yields that are averaged over years
        sim_yields = []

        ## list for german wide means
        mean_ts1 = []
        ## list for mean of landkreise averages
        mean_ts2 = []
    
        for year in year_lst:
            print(year, crop)
            ## list all files of the current run
            search_term = rf'{yield_pth}/{crop_name}_Yield_{year}*.asc'
            file_pth = glob.glob(search_term)
            if file_pth:
                file_pth = file_pth[0]
            else:
                continue

            print(file_pth)

            ## Open simulated crop yields
            yield_ras = gdal.Open(file_pth)
            yield_arr = yield_ras.ReadAsArray()
            ndv_yields = yield_ras.GetRasterBand(1).GetNoDataValue()

            ## Create a mask that sets all no data values to 0
            ndv_mask = np.where(yield_arr == ndv_yields, 0, 1)

            ## Open crop mask aggregated
            crop_ras = gdal.Open(CROP_MSK_PTH2.format(crop))
            crop_arr = crop_ras.ReadAsArray()

            ## Create crop mask from crop array and ndv array
            mask = np.where(ndv_mask == 1, crop_arr, 0)

            ## Mask yield array with crop masked
            yield_arr_m = np.ma.masked_where(mask == 0, yield_arr)

            ## Convert dry matter to fresh matter
            ## !! Not necessary anymore, as reference data were already converted from fresh matter to dry matter !!
            # yield_arr_m = (yield_arr_m/conv_factor)*100

            yield_arr_lst.append(yield_arr_m.copy())

            ## Save mean german yield of current crop and year to list
            mean_yield = np.ma.mean(yield_arr_m)
            mean_ts1.append([crop, year, mean_yield])

        if not yield_arr_lst:
            print(f"There are no rasters for {crop}. Run ID: {run_id}")
            continue

        ## Average yields over all years
        yields_aggr = np.ma.mean(yield_arr_lst, axis=0)

        for fid in ldk_ids:
            ## Create landkreis-specific mask
            mask = np.where(ldk_arr == fid, 1, 0)

            ## Extract average yield for each landkreis across all years
            yields_aggr_m = np.ma.masked_where(mask == 0, yields_aggr)
            yields_aggr_m = np.ma.compressed(yields_aggr_m)
            sim_yield = np.mean(yields_aggr_m)
            sim_yields.append([fid, crop, sim_yield])

            ## Extract average yield for each landkreis and year separately
            for y, yield_arr in enumerate(yield_arr_lst):
                yield_arr_m = np.ma.masked_where(mask == 0, yield_arr)
                yield_arr_m = np.ma.compressed(yield_arr_m)
                mean_yield = np.ma.mean(yield_arr_m)
                year = year_lst[y]
                mean_ts2.append([crop, year, fid, mean_yield])

        ## Create df with mean yields of Germany
        cols = ['Crop', 'Year', 'mean_yield']
        df_sim_ger = pd.DataFrame(mean_ts1)
        df_sim_ger.columns = cols

        ## Create df with mean yields of landkreise averaged over all years
        cols = ['ID', 'Crop', 'yield_sim']
        df_sim = pd.DataFrame(sim_yields)
        df_sim.columns = cols

        ## Create df with mean yields of landkreises for each year separetely
        cols = ['Crop', 'Year', 'ID', 'mean_yield']
        df_sim_ldk = pd.DataFrame(mean_ts2)
        df_sim_ldk.columns = cols

        ## Open observed yields landkreis and for germany
        df_obs_l = pd.read_csv(OBS_PTH_LDK)
        df_obs_l[YIELD_OBS_COL] = df_obs_l[YIELD_OBS_COL] * 100
        df_obs_ger = pd.read_csv(OBS_PTH_GER)
        df_obs_ger[YIELD_OBS_COL] = df_obs_ger[YIELD_OBS_COL] * 100

        ## Change dtype of merge columns, so that merging works
        df_obs_l[YIELD_OBS_COL] = df_obs_l[YIELD_OBS_COL].apply(float)
        df_obs_l.Crop = df_obs_l.Crop.apply(str)
        df_obs_l.ID = df_obs_l.ID.apply(int)
        df_sim.Crop = df_sim.Crop.apply(str)
        df_sim.ID = df_sim.ID.apply(int)

        ## Subset reference dfs to period of interest
        min_year = min(year_lst)
        max_year = max(year_lst)
        df_obs_l = df_obs_l.loc[(df_obs_l.Year >= min_year) & (df_obs_l.Year <= max_year)]
        df_obs_ger = df_obs_ger.loc[(df_obs_ger.Year >= min_year) & (df_obs_ger.Year <= max_year)]

        #### For validation on Landkreise-level
        ## Average the yield over all years per landkreis
        df_obs_aggr = df_obs_l.groupby(['ID', 'Crop'])[YIELD_OBS_COL].mean().reset_index()

        ## Merge simulated with observed values
        df_val = pd.merge(df_sim, df_obs_aggr, how='left', on=['ID', 'Crop'])
        out_pth = rf"{OUT_FOLDER_TABLES}/{crop}_nuts3_GER_sim-obs_yields_years_aggr_{run_id}.csv"
        df_val.to_csv(out_pth, index=False)

        #### For validation on German-wide-level
        ## Average the yields over all landkreise per year just for comparison 
        ## (df with time series of mean mean yields of landkreise averages)
        df_sim_ger2 = df_sim_ldk.groupby(['Crop', 'Year'])['mean_yield'].mean().reset_index()

        ## Combine simulated yearly mean yields per (german-wide) with yearly mean yields of landkreise averages and
        ## observed yearly mean yields (landkreise averages)
        df_obs_ger['Year'] = df_obs_ger['Year'].astype(int)
        df_ts_ger = pd.merge(df_obs_ger, df_sim_ger, how='left', on=['Crop', 'Year'])
        df_ts_ger = pd.merge(df_ts_ger, df_sim_ger2, how='left', on=['Crop', 'Year'])
        df_ts_ger.columns = ['Crop', 'Crop_name_de', 'Year', 'yield_obs', 'yield_obs_detr', 'yield_sim_geravg', 'yield_sim_ldkavg']
        out_pth = rf"{OUT_FOLDER_TABLES}/{crop}_GER_sim-obs_yields_years_ts_{run_id}.csv"
        df_ts_ger.to_csv(out_pth, index=False)

        ## Combine time series of observed and simulated mean yields per landkreis
        df_obs_l['Year'] = df_obs_l['Year'].astype(int)
        df_ts_ldk = pd.merge(df_obs_l, df_sim_ldk, how='left', on=['Crop', 'Year', 'ID'])
        out_pth = rf"{OUT_FOLDER_TABLES}/{crop}_nuts3_GER_sim_vs_obs_yields_years_ts_{run_id}.csv"
        df_ts_ldk.to_csv(out_pth, index=False)


if __name__ == '__main__':
    merge_observation_simulation_years_aggregated()
