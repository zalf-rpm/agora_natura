import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import geopandas as gpd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D

wd = r"D:\projects\AgoraNatura"
os.chdir(wd)

## Observations for WW, WRye, Wrape, WB, SM
OBS_PTH = rf"D:\projects\AgoraNatura\tables\official yield statistics\nuts3_ertraege_D_05_crops_1990-2019.csv"

## Observations for 'SB', 'OAT', 'TR', 'PO', 'SU'
OBS_PTH2 = r"D:\projects\AgoraNatura\tables\official yield statistics\nuts3_ertraege_D_10_crops_1999-2019.csv"

## Observations for Luzerne
OBS_PTH_CLALF = r"D:\projects\AgoraNatura\tables\official yield statistics\Luzerne_Kleegrasgemisch_NUTS.csv"


OUT_FOLDER = rf"D:\projects\AgoraNatura\tables\official yield statistics"

## [Range of years to be used, conversion factor fresh to dry matter]
## (For OAT and TR just used the same converstion factor as for other cereals)
CROP_DICT = {
    'WW': [[1999, 2019], 86],
    'SM': [[1999, 2019], 32],
    'WB': [[1999, 2019], 86],
    'WR': [[1999, 2019], 91],
    'WRye': [[1999, 2019], 86],
    'PO': [[1999, 2019], 22.5],
    'SB': [[1999, 2019], 86],
    'SU': [[1999, 2019], 22.5],
    'OAT': [[1999, 2019], 86],
    'TR': [[1999, 2019], 86],
    'CLALF': [[1999, 2019], 100],

}

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def create_folder(directory):
    import os
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' + directory )

def prepare_reference_df():

    ## Transform df from Anna Hampf into long format
    ## Here statistics on potatoe and sugar-beet are missing
    df_obs = pd.read_csv(OBS_PTH)
    df_obs = df_obs.drop(columns=['Nuts1_Bez', 'Nuts3_ID', 'Kreise', 'Nuts3_Bez'])
    value_vars = list(df_obs.columns)
    value_vars.remove('ID')
    df_obs_l = pd.melt(df_obs, id_vars='ID', value_vars=value_vars, value_name='yield_obs')
    df_obs_l[['Crop', 'Year']] = df_obs_l['variable'].str.split('_', 1, expand=True)
    crop_dict = {'WW':'WW', 'WG':'WB', 'WR':'WRye', 'WRa':'WR', 'SM':'SM'}
    df_obs_l['Crop'] = df_obs_l['Crop'].map(crop_dict)
    df_obs_l = df_obs_l.dropna(subset=['yield_obs'])
    df_obs_l['yield_obs'] = df_obs_l['yield_obs'] * 100

    ## Add additional crops
    df_obs_ext = pd.read_csv(OBS_PTH2)
    value_vars = ['SB', 'OAT', 'TR', 'PO', 'SU']
    df_obs_l2 = pd.melt(df_obs_ext, id_vars=['ID', 'Year'], value_vars=value_vars, value_name='yield_obs')
    df_obs_l2 = df_obs_l2.dropna()
    df_obs_l2['yield_obs'] = df_obs_l2['yield_obs'].astype(float)
    df_obs_l2['yield_obs'] = df_obs_l2['yield_obs'] * 100
    df_obs_l2.columns = ['ID', 'Year', 'Crop', 'yield_obs']

    ## Drop rows with Dauergründland
    indeces = df_obs_l2[df_obs_l2['ID'] == 'DG'].index
    df_obs_l2.drop(indeces, inplace=True)

    ## Add Luzerne
    df_obs_clalf = pd.read_csv(OBS_PTH_CLALF, encoding='ISO-8859-1', sep=';')
    cols = [f"CLALF_{year}" for year in range(1990, 2020)]
    cols.insert(0, "ID")
    df_obs_clalf = df_obs_clalf[cols]
    new_cols = [f"{year}" for year in range(1990, 2020)]
    new_cols.insert(0, "ID")
    df_obs_clalf.columns = new_cols
    new_cols.remove("ID")
    for col in new_cols:
        df_obs_clalf[col] = df_obs_clalf[col].str.replace(",", ".", regex=False)
        df_obs_clalf[col] = df_obs_clalf[col].str.replace(" ", "", regex=False)
        df_obs_clalf[col] = df_obs_clalf[col].str.replace("(", "", regex=False)
        df_obs_clalf[col] = df_obs_clalf[col].str.replace(")", "", regex=False)
        df_obs_clalf.loc[df_obs_clalf[col] == "NA", col] = None
        df_obs_clalf.loc[df_obs_clalf[col] == "NA.", col] = None
        # df_obs_clalf.loc[df_obs_clalf[col].isna(), col] = None
        df_obs_clalf[col] = df_obs_clalf[col].astype(float)
    df_obs_clalf = df_obs_clalf.loc[df_obs_clalf["ID"].notna()].copy()
    df_obs_clalf["ID"] = df_obs_clalf["ID"].astype(int)
    df_obs_clalf["ID"] = df_obs_clalf["ID"].astype(str)
    df_obs_clalf2 = pd.melt(df_obs_clalf, id_vars=['ID'], value_vars=new_cols, value_name='yield_obs')
    df_obs_clalf2["Crop"] = "CLALF"
    df_obs_clalf2.rename(columns={"variable": "Year"}, inplace=True)
    df_obs_clalf2 = df_obs_clalf2[['ID', 'Year', 'Crop', 'yield_obs']]
    df_obs_clalf2 = df_obs_clalf2.dropna()
    df_obs_clalf2['yield_obs'] = df_obs_clalf2['yield_obs'] * 100

    df_obs_clalf[new_cols].mean().tolist()

    ## Transform data types
    df_obs_l2['ID'] = df_obs_l2['ID'].astype(int)
    df_obs_l2['Crop'] = df_obs_l2['Crop'].astype(str)
    df_obs_l2['Year'] = df_obs_l2['Year'].astype(int)
    df_obs_l2['yield_obs'] = df_obs_l2['yield_obs'].astype(float)

    df_obs_l['ID'] = df_obs_l['ID'].astype(int)
    df_obs_l['Crop'] = df_obs_l['Crop'].astype(str)
    df_obs_l['Year'] = df_obs_l['Year'].astype(int)
    df_obs_l['yield_obs'] = df_obs_l['yield_obs'].astype(float)

    ## Concat all tables
    df_obs_l = pd.concat([df_obs_l[['ID', 'Crop', 'Year', 'yield_obs']],
                          df_obs_l2[['ID', 'Crop', 'Year', 'yield_obs']],
                          df_obs_clalf2[['ID', 'Crop', 'Year', 'yield_obs']]])

    pth = rf"{OUT_FOLDER}\nuts3_ertraege_D_11-crops_1990-2019_long-format.csv"
    df_obs_l.to_csv(pth)

    df_obs_clalf2["Crop_name_de"] = "Luzerne"
    df_obs_clalf2 = df_obs_clalf2[["ID","Crop_name_de","Crop","Year","yield_obs"]]
    df_obs_clalf2['yield_obs'] = df_obs_clalf2['yield_obs'] / 100
    pth = rf"{OUT_FOLDER}\nuts3_ertraege_D_CLALF_1990-2019_long-format.csv"
    df_obs_clalf2.to_csv(pth, index=False)
    print()

def detrend_yield_observations_germany():

    #### Yields for entire Germany
    pth = rf"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Deutschland_11_Frucharten_1990-2020.csv"

    df = pd.read_csv(pth)
    crops = df['Crop'].unique()

    df_lst = []
    out_lst = []

    for crop in crops:
        # crop = 'WW'
        year_range = CROP_DICT[crop][0]
        min_year = year_range[0]
        max_year = year_range[1]

        df_sub = df.loc[df['Crop'] == crop]
        df_sub = df_sub.sort_values(by='Year', ascending=True)
        df_sub = df_sub.loc[(df_sub['Year'] >= min_year)].copy()
        df_sub = df_sub.loc[(df_sub['Year'] <= max_year)].copy()

        if len(df_sub) <= 2:
            continue

        # df_sub.plot(y='yield_obs', x='Crop')
        # plt.show()

        x = np.array(df_sub['Year'])  # np.arange(0, len(df_sub))
        x = np.reshape(x, (len(x), 1))
        y = np.array(df_sub['yield_obs'])

        model = LinearRegression()
        model.fit(x, y)

        trend = model.predict(x)
        trend_coef = model.coef_[0]
        if trend_coef > 0:
            start = trend[0]
            detrended = [y[i] - (trend[i] -start) for i in range(0, len(y))]
            print(crop, model.coef_[0], "--> detrending")
        else:
            print(crop, model.coef_[0], "--> no detrending")
            detrended = df_sub['yield_obs'].tolist()

        df_sub['yield_obs_detr'] = detrended

        ## Convert fresh matter to dry matter
        conv_factor = CROP_DICT[crop][1]
        df_sub['yield_obs'] = (df_sub['yield_obs'] / 100) * conv_factor
        df_sub['yield_obs_detr'] = (df_sub['yield_obs_detr'] / 100 ) * conv_factor

        df_lst.append(df_sub)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(df_sub['Year'], df_sub['yield_obs'])
        ax.plot(df_sub['Year'], df_sub['yield_obs_detr'])
        ax.plot(df_sub['Year'], trend)
        ax.annotate(text=f"Trend: {round(trend_coef, 3)}", xy=(0.05, 0.85), xycoords='axes fraction')
        pth = fr"D:\projects\AgoraNatura\figures\detrending_yield_obs\german_wide\{crop}_detrended_ts.png"
        plt.savefig(pth)
        # plt.show()
        plt.close()

        out_lst.append([crop, trend_coef])

    df_trends = pd.DataFrame(out_lst)
    df_trends.columns = ['Crop', 'trend_coeff']
    out_pth = rf"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Deutschland_11_Frucharten_1999-2019_trend_coefficents.csv"
    df_trends.to_csv(out_pth, index=False)

    df_out = pd.concat(df_lst)
    out_pth = rf"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Deutschland_11_Frucharten_1999-2019_detrended.csv"
    df_out.to_csv(out_pth, index=False)

def detrend_yield_observations_landkreise():
    #### Yields for Landkreise
    pth = rf"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Landkreise_11_Frucharten_1999-2019.csv"

    df = pd.read_csv(pth)
    crops = df['Crop'].unique()
    districts = df['ID'].unique()
    print(f'Number of districts: {len(districts)}')

    df_lst = []
    out_lst = []

    for crop in crops:
        for distr in districts:
            # crop = 'WW'
            year_range = CROP_DICT[crop][0]
            min_year = year_range[0]
            max_year = year_range[1]

            df_sub = df.loc[df['Crop'] == crop]
            df_sub = df_sub.loc[df_sub['ID'] == distr]
            df_sub = df_sub.sort_values(by='Year', ascending=True)
            df_sub = df_sub.loc[(df_sub['Year'] >= min_year)].copy()
            df_sub = df_sub.loc[(df_sub['Year'] <= max_year)].copy()

            if len(df_sub) <= 2:
                continue

            # df_sub.plot(y='yield_obs', x='Crop')
            # plt.show()

            x = x = np.array(df_sub['Year'])  #np.arange(0, len(df_sub))
            x = np.reshape(x, (len(x), 1))
            y = np.array(df_sub['yield_obs'])

            model = LinearRegression()
            model.fit(x, y)

            trend = model.predict(x)
            trend_coef = model.coef_[0]
            if trend_coef > 0:
                start = trend[0]
                detrended = [y[i] - (trend[i] - start) for i in range(0, len(y))]
                print(distr, crop, model.coef_[0], "--> detrending")
            else:
                print(distr, crop, model.coef_[0], "--> no detrending")
                detrended = df_sub['yield_obs'].tolist()

            df_sub['yield_obs_detr'] = detrended

            ## Convert fresh matter to dry matter
            conv_factor = CROP_DICT[crop][1]
            df_sub['yield_obs'] = (df_sub['yield_obs'] / 100) * conv_factor
            df_sub['yield_obs_detr'] = (df_sub['yield_obs_detr'] / 100) * conv_factor

            df_lst.append(df_sub)

            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.plot(df_sub['Year'], df_sub['yield_obs'])
            ax.plot(df_sub['Year'], df_sub['yield_obs_detr'])
            ax.plot(df_sub['Year'], trend)
            ax.annotate(text=f"Trend: {round(trend_coef, 3)}", xy=(0.05, 0.85), xycoords='axes fraction')
            out_folder = fr"D:\projects\AgoraNatura\figures\detrending_yield_obs\district_level\{crop}"
            create_folder(out_folder)
            pth = fr"{out_folder}\{distr}_{crop}_detrended_ts.png"
            plt.savefig(pth)
            plt.close()

            out_lst.append([distr, crop, trend_coef])

    df_trends = pd.DataFrame(out_lst)
    df_trends.columns = ['ID', 'Crop', 'trend_coeff']
    out_pth = rf"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Landkreise_11_Frucharten_1999-2019_trend_coefficents.csv"
    df_trends.to_csv(out_pth, index=False)

    df_out = pd.concat(df_lst)
    out_pth = rf"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Landkreise_11_Frucharten_1999-2019_detrended.csv"
    df_out.to_csv(out_pth, index=False)

def plot_trend_maps():
    gdf = gpd.read_file(r"D:\projects\AgoraNatura\vector\administrative\GER_landkreise_25832.shp")
    df_trends = pd.read_csv(rf"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Landkreise_11_Frucharten_1999-2019_trend_coefficents.csv")

    crops = df_trends['Crop'].unique()

    gdf['cca_2'] = gdf['cca_2'].astype(int)

    for crop in crops:
        df_sub = df_trends.loc[df_trends['Crop'] == crop]
        shp = pd.merge(gdf, df_sub, how='left', left_on='cca_2', right_on='ID')

        font_size = 12
        plt.rcParams['legend.title_fontsize'] = f'{font_size}'
        plt.rcParams["font.family"] = "Calibri"

        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=cm2inch(20.0, 10))

        divider = make_axes_locatable(axs)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        max_val = max(shp['trend_coeff'].max(), abs(shp['trend_coeff'].min()))
        # max_val = 10

        # shp.plot(column=diff_col, ax=axs, legend=True)
        shp.plot(column='trend_coeff', ax=axs, vmin=-max_val, vmax=max_val, legend=True, cax=cax,
                 legend_kwds={'label': "Yield difference [kg/ha]"}, cmap='RdYlBu')

        axs.set_title(f'Trends {crop} 1999-2020')

        axs.set_axis_off()

        # fig.suptitle(title)

        # plt.show()
        out_folder = fr"D:\projects\AgoraNatura\figures\detrending_yield_obs\district_level_maps"
        create_folder(out_folder)
        out_pth = fr"{out_folder}\{crop}_trend_coefficients.png"
        plt.tight_layout()
        plt.savefig(out_pth)
        plt.close()

def trend_coefficients_comparison():
    pth = r"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Landkreise_11_Frucharten_1999-2019_detrended.csv"
    pth_ger = r"D:\projects\AgoraNatura\tables\official yield statistics\Ertraege_Deutschland_11_Frucharten_1999-2019_detrended.csv"

    df = pd.read_csv(pth)
    df_ger = pd.read_csv(pth_ger)

    df_summ = df.groupby(["Crop", "Year"])[["yield_obs", "yield_obs_detr"]].mean().reset_index()
    df_summ.columns = ["Crop", 'Year', 'yield_obs_ldk', 'yield_obs_detr_ldk']

    df_plt = pd.merge(df_ger, df_summ, how='left', on=["Crop", "Year"])
    crops = df_plt['Crop'].unique()

    for crop in crops:
        df_sub = df_plt.loc[df_plt['Crop'] == crop]

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(df_sub['Year'], df_sub['yield_obs'], linestyle='dashed', color='red')
        ax.plot(df_sub['Year'], df_sub['yield_obs_detr'], linestyle='solid', color='red')
        ax.plot(df_sub['Year'], df_sub['yield_obs_ldk'], linestyle='dashed', color='blue')
        ax.plot(df_sub['Year'], df_sub['yield_obs_detr_ldk'], linestyle='solid', color='blue')
        custom_lines = [Line2D([0], [0], color='red', lw=1, linestyle='dashed'),
                        Line2D([0], [0], color='red', lw=1, linestyle='solid'),
                        Line2D([0], [0], color='blue', lw=1, linestyle='dashed'),
                        Line2D([0], [0], color='blue', lw=1, linestyle='solid'),
                        ]
        legend_labels = ["Ger-orig", "Ger-detr", "Ldk_aggr-orig", "Ldk_aggr-orig"]
        ax.legend(custom_lines, legend_labels, loc='upper left')

        fig.suptitle(f'comparison orig - detr & ger - ldk_aggr 1999-2019')

        out_folder = fr"D:\projects\AgoraNatura\figures\detrending_yield_obs\comparison"
        create_folder(out_folder)
        pth = fr"{out_folder}\{crop}_comparison_ts.png"
        plt.savefig(pth)
        plt.close()

if __name__ == '__main__':
    prepare_reference_df()
    detrend_yield_observations_germany()
    detrend_yield_observations_landkreise()
    plot_trend_maps()
    trend_coefficients_comparison()