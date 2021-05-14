# TBD - Image export
# TBD - Expand main sequence
# TBD - Finish labeling data
# TBD - Comment more sections of code for clarity

# Imports

# Import as
import pandas as pd
import plotly.graph_objs as go
import numpy as np

# Import from
from plotly.subplots import make_subplots
from scipy import stats

# PATH variables (Main path and subfolders)
PATH = "./Data"
STATSPATH = "/Statistical data/"
ALLPATH = "/All waves/"
SPECPATH = "/Spectral data/"

# Plotting functions


def f_ff(stent):
    """ Frequency plotting

    Arguments:
    stent -- entry file
    """
    df = pd.read_csv(PATH + SPECPATH + stent, usecols=np.arange(261))  # Rotate the data
    # print(df.head().to_string())
    fig = make_subplots(rows=1, cols=1)

    fig.append_trace(go.Scatter(
        x=df['date2'],
        y=df['F 0'],
        name="F 0"),
        row=1,
        col=1
    )

    fig.append_trace(go.Scatter(
        x=df['date2'],
        y=df['F 0.5'],
        name="F 0.5"),
        row=1,
        col=1
    )

    fig.append_trace(go.Scatter(
        x=df['date2'],
        y=df['F 1'],
        name="F 1"),
        row=1,
        col=1
    )

    fig.update_layout(
        title_text="Frecuencia y tiempo",
        title_x=0.5,
        font=dict(size=18))
    fig.update_xaxes(title_text="Fecha")
    fig.update_yaxes(title_text="Frecuencia")

    fig.show()


def gen_scat(stent, stpath, vart, var1, *argv):
    """ Generic scatter plot

    Arguments:
    stent -- entry file
    stpath -- path variable
    var1 -- first variable (x-axis)
    vart -- y axis
    args -- additional variables to plot (x-ax)
    """
    df = pd.read_csv(PATH + stpath + stent)
    # print(df.head().to_string())
    fig = make_subplots(rows=1, cols=1)

    df['dateT'] = pd.to_datetime(df['date2'])

    # date adjustment
    df = df_tf(df, 2019, 6, 15, 2019, 10, 16)
    df = df.sort_values(by=['dateT'])
    # df = remove_outliers(df, var1)

    fig.append_trace(go.Scatter(
        x=df[vart],
        y=df[var1],
        name=var1),
        row=1,
        col=1
    )

    for arg in argv:
        fig.append_trace(go.Scatter(
            x=df[vart],
            y=df[arg],
            name=arg),
            row=1,
            col=1
        )

    fig.update_layout(
        title_text=(var1 + " y " + vart),
        title_x=0.5,
        font=dict(size=18))

    fig.show()


def gen_vs(stent, stpath, varx, vary):
    """ Generic VS-plot for x-vs-y arrangements.

    Arguments:
    stent -- entry file
    stpath -- path variable
    varx -- variable x
    vary -- variable y
    """
    df = pd.read_csv(PATH + stpath + stent)
    # print(df.head().to_string())
    fig = make_subplots(rows=1, cols=1)

    fig.append_trace(go.Scatter(
        x=df[varx],
        y=df[vary],
        name=(varx + " " + vary)),
        row=1,
        col=1
    )

    fig.update_layout(
        title_text=(varx + " y " + vary),
        title_x=0.5,
        font=dict(size=18))

    fig.show()


def df_tf(df, d1, m1, y1, d2, m2, y2):
    """ DF timeframe setter

    Arguments:
    df -- dataframe
    d1 -- start day
    m1 -- start month
    y1 -- start year
    d2 -- end day
    m2 -- end month
    y2 -- end year
    """
    maska = (str(d1) + '/' + str(m1) + '/' + str(y1) + ' 00:00:00')
    maskb = (str(d2) + '/' + str(m2) + '/' + str(y2) + ' 00:00:00')
    mask = (df['dateT'] > maska) & (df['dateT'] <= maskb)
    return df[mask]


def remove_outliers(df, col):
    """ removes outliers (+- 3sd's)

    Arguments:
    df -- dataframe
    col -- column to work on
    """
    df = df[(np.abs(stats.zscore(df[col])) < 0.5)]
    return df


# Main sequence
gen_scat("boya_32_statistical_params_20191119142107.csv", STATSPATH, 'dateT', "Energy", "Power")
gen_scat("boya_32_statistical_params_20191119142107.csv", STATSPATH, 'dateT', "Tp", "Hm0", 'DirTp', 'Hmax')

# gen_scat("boya_32_all_waves_20191125111628_agosto.csv", ALLPATH, "direction")
# gen_scat("boya_32_statistical_params_20191119142107.csv", STATSPATH, "T01", "T02")
# gen_scat("boya_32_buoy_params_20191119142048.csv", STATSPATH,
# "temperatura_agua", "temperatura_interna", varT="fecha_gps2")
# f_ff("boya_32_spectral_param_psd_20191125112048_junio.csv")
