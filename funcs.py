# TBD - Image export
# TBD - Expand main sequence
# TBD - Finish labeling data
# TBD - Comment more sections of code for clarity

# Import
import os
import plotly

# Import as
import pandas as pd
import plotly.graph_objs as go
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Import from
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression

plotly.io.orca.config.executable = 'C:/Users/RRX\AppData\Local\Programs\orca/orca.exe'
pd.options.mode.chained_assignment = None  # default='warn'

# PATH variables (Main path and subfolders)
PATH = "./Data/"
PATHX = "./Output/"
# Plotting functions

def to_df(stent, sheet):
    df = pd.read_excel(PATH + stent, parse_dates=['Date'], nrows=98241, sheet_name=sheet)
    return df


def gen_scat(df, name, stmo, endmo, vart, *argv):
    """ Generic scatter plot

    Arguments:
    df -- entry df
    stpath -- path variable
    var1 -- first variable (x-axis)
    vart -- y axis
    args -- additional variables to plot (x-ax)
    """
    # df = pd.read_excel(PATH + stent)
    # print(df.head().to_string())
    fig = make_subplots(rows=1, cols=1)

    df[vart] = pd.to_datetime(df['Date']).dt.date
    df[vart] = df[vart].astype(str) + ' ' + df['UTC Time'].astype(str)
    # df[vart] = df[vart][0:98241]
    df[vart] = pd.to_datetime(df[vart])

    # date adjustment
    df = df_tf(df, vart, 2019, stmo, 1, 2019, endmo, 1)
    df = df.sort_values(by=[vart])
    #df = remove_outliers(df, var1)


    for arg in argv:
        fig.append_trace(go.Scatter(
            x=df[vart],
            y=df[arg],
            name=arg),
            row=1,
            col=1
        )
        rolling = df[arg].rolling(1080).mean()
        fig.append_trace(go.Scatter(
            x=df[vart],
            y=rolling,
            name=arg+"(P. 1D)"),
            row=1,
            col=1
        )
        # rolling = df[arg].rolling(1080*3).mean()
        # fig.append_trace(go.Scatter(
        #     x=df[vart],
        #     y=rolling,
        #     name=arg + "(3da)"),
        #     row=1,
        #     col=1
        # )
        rolling = df[arg].rolling(1080*6).mean()
        fig.append_trace(go.Scatter(
            x=df[vart],
            y=rolling,
            name=arg + "(P. 6D)"),
            row=1,
            col=1
        )

    fig.update_layout(
        title_text=(" "),
        title_x=0.5,
        font=dict(size=18))

    # fig.show()
    if not os.path.exists(PATHX + name + "/"):
        os.makedirs(PATHX + name + "/")
    fig.write_image(PATHX + name + "/avg_plot" + argv[0] + ".png", width=1250, height=750)


def df_tf(df, vart, d1, m1, y1, d2, m2, y2):
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
    m1 = int(m1)
    m2 = int(m2)
    maska = (str(d1) + '/' + str(m1) + '/' + str(y1) + ' 00:00:00')
    maskb = (str(d2) + '/' + str(m2) + '/' + str(y2) + ' 00:00:00')
    mask = (df[vart] > maska) & (df[vart] <= maskb)
    return df[mask]


def remove_outliers(df, col):
    """ removes outliers (+- 3sd's)

    Arguments:
    df -- dataframe
    col -- column to work on
    """
    df = df[(np.abs(stats.zscore(df[col])) < 0.5)]
    return df


def info(df, *argv):
    for arg in argv:
        print(arg)
        print("Mean: {0}".format(df[arg].mean()))
        print("Std. Dev: {0}".format(df[arg].std()))


def num2dir(d,deruta16=True):
    if deruta16: dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    else: dirs = ['N', 'NNE', 'ENE', 'E', 'ESE', 'SSE', 'S', 'SSW', 'WSW', 'W', 'WNW', 'NNW']
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]


def makerange(min, max, nbins):
    ranges = []
    rangestx = []
    rangelen = (max - min) / (nbins)
    rangeval = min
    ranges.append(rangeval)

    while rangeval < max:
        rangeval2 = rangeval + rangelen
        rangeval2 = round(rangeval2, 2)
        rangeval = round(rangeval, 2)
        rangestx.append("{0} - {1}".format(rangeval, rangeval2))
        ranges.append(rangeval2)
        rangeval = round(rangeval2, 2)

    return rangestx, ranges


def num2range(d, rangestx, ranges, index=False):
    lx = 0
    while d > ranges[lx]:
        try:
            if d < ranges[lx+1]:
                break
        except:
            0
        lx += 1
    if index:
        return lx
    else:
        return rangestx[lx]


def wrose(dfa, name, speed, dir, n_bins):
    df = dfa.filter([speed, dir],axis=1)
    df = flipdir(df, dir)
    # df = DataF.dropna()

    rangestx, ranges = makerange(df[speed].min(), df[speed].max(), n_bins)
    df['Indexes'] = df[speed].apply(lambda x: num2range(x, rangestx, ranges, index=True))
    df['Interval (cm/s)'] = df[speed].apply(lambda x: num2range(x, rangestx, ranges))
    df['Direction'] = df[dir].apply(lambda x: num2dir(x))

    df = df.groupby(["Indexes", "Direction", "Interval (cm/s)"]).size().reset_index(name="frequency")
    df = df.sort_values(by='Indexes')
    dirdict = {"Direction": ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W',
                             'WNW', 'NW', 'NNW']}

    fig = px.bar_polar(df, r='frequency', theta="Direction",
                       color="Interval (cm/s)", template="plotly_white",
                       color_discrete_sequence=px.colors.sequential.Plasma_r,
                       category_orders=dirdict,
                       title="Rosa de viento: {}".format(dfa["Date"].values[0])
                       )
    # fig.show()

    if not os.path.exists(PATHX + name + "/"):
        os.makedirs(PATHX + name + "/")
    fig.write_image(PATHX + name + "/wrose.png", width=1250, height=750)


def gen_vs(stent, name, varx, vary):
    """ Generic VS-plot for x-vs-y arrangements.

    Arguments:
    stent -- entry file
    stpath -- path variable
    varx -- variable x
    vary -- variable y
    """
    df = stent

    fig = make_subplots(rows=1, cols=1)

    X = df[varx].values.reshape(-1,1)

    model = LinearRegression()
    model.fit(X, df[vary])

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1, 1))

    fig = px.scatter(df, x=varx, y=vary, opacity=0.65)
    fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))

    # fig.show()
    if not os.path.exists(PATHX + name + "/"):
        os.makedirs(PATHX + name + "/")
    fig.write_image(PATHX + name + "/gen_vs.png", width=1250, height=750)


def sep_df(df, column):
    df.loc[:,'Month'] = df[column].dt.month
    #print(df)
    array = df['Month'].unique()
    frames = []
    for object in array:
        df_new = df.loc[df['Month'] == object]
        df_new.loc[:,"Date"] = df_new["Date"].dt.date
        #print(df_new)
        frames.append(df_new)
    #print(frames)
    return frames


def box_plot(df, name, date, *argv):
    df = grpday_df(df)
    fig = make_subplots(rows=1, cols=1)
    for arg in argv:
        fig.append_trace(go.Box(x=df[date], y=df[arg], name=arg), row=1, col=1)

    fig.update_layout(title_text="Distribución de datos: día {}".format(df[date].values[0]),
                      title_x=0.5,
                      font=dict(size=18),
                      yaxis_title_text='magnitud',  # xaxis label
                      xaxis_title_text='día',  # yaxis label
                      bargap=0.0  # gap between bars of the same location coordinates
                      )
    # fig.show()
    if not os.path.exists(PATHX + name + "/"):
        os.makedirs(PATHX + name + "/")
    fig.write_image(PATHX + name + "/box_plot_" + argv[0] +".png", width=1250, height=750)



def grpday_df(df):
    df = df.copy(deep=True)
    df.loc[:, "Date"] = df["Date"].dt.date
    return df


def flipdir(df, dir):
    df[dir] = df[dir] + 180
    return df


def avg_plot(df, name, date, *argv):
    dfa = df.copy(deep=True)
    dfa = dfa.resample('H', on=date).mean()
    # print(dfa)
    fig = make_subplots(rows=1, cols=1)
    for arg in argv:
        fig.append_trace(go.Scatter(x=dfa.index, y=dfa[arg], name=arg), row=1, col=1)

    fig.update_layout(title_text="Distribución de datos: día",
                      title_x=0.5,
                      font=dict(size=18),
                      yaxis_title_text='magnitud',  # xaxis label
                      xaxis_title_text='día',  # yaxis label
                      bargap=0.0  # gap between bars of the same location coordinates
                      )
    # fig.show()

    if not os.path.exists(PATHX + name + "/"):
        os.makedirs(PATHX + name + "/")
    fig.write_image(PATHX + name + "/avg_plot.png", width=1250, height=750)


def heat(df, name):
    matrix = np.triu(df.corr())
    sns.heatmap(df.corr(), annot=True, fmt='.1g', mask=matrix)
    # plt.show()

    if not os.path.exists(PATHX + name + "/"):
        os.makedirs(PATHX + name + "/")
    plt.tight_layout()
    plt.savefig(PATHX + name + "/heatmap.png")
    plt.clf()


def compare(dfs, stmo, endmo, arg):
    fig = make_subplots(rows=1, cols=1)
    i = 0
    for df in dfs:
        dfa = df.resample('D', on="Date").mean()
        i += 1
        # date adjustment
        # df = df_tf(df, dfa.index, 2019, stmo, 1, 2019, endmo, 1)
        # df = df.sort_values(by=[dfa.index])
        # df = remove_outliers(df, var1)

        fig.append_trace(go.Scatter(
            x=dfa.index,
            y=dfa[arg],
            name="{0}, Turbina {1}".format(arg,i)),
            row=1,
            col=1
        )


    fig.update_layout(
        title_text=(" "),
        title_x=0.5,
        font=dict(size=18))

    fig.show()
    # if not os.path.exists(PATHX + name + "/"):
    #     os.makedirs(PATHX + name + "/")
    # fig.write_image(PATHX + name + "/avg_plot" + argv[0] + ".png", width=1250, height=750)
    # return 0


# dfa = df.copy(deep=True)
# dfa = dfa.resample('D', on="Date").mean()
# dfa = dfa.sort_values(by=varx)
# # print(df.head().to_string())