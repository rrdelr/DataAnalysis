# Imports
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from matplotlib import pyplot as plt
from windrose import WindroseAxes
import plotly.express as px

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
        rangestx.append("{0} - {1}".format(rangeval, rangeval2))
        ranges.append(rangeval2)
        rangeval = round(rangeval2, 2)

    print(rangestx)
    print(ranges)

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


# PATH variables (Main path and subfolders)
PATH = "./Data/"

# Code

def wrose(stent, speed, dir, n_bins):
    stent = pd.read_excel(PATH + stent)
    stent = stent[0:98241]
    df = stent.filter([speed, dir],axis=1)
    # df = DataF.dropna()

    rangestx, ranges = makerange(df[speed].min(), df[speed].max(), n_bins)
    df['Indexes'] = df[speed].apply(lambda x: num2range(x, rangestx, ranges, index=True))
    df['Interval (cm/s)'] = df[speed].apply(lambda x: num2range(x, rangestx, ranges))
    df['Direction'] = df[dir].apply(lambda x: num2dir(x))

    df = df.groupby(["Indexes", "Direction", "Interval (cm/s)"]).size().reset_index(name="frequency")
    df = df.sort_values(by='Indexes')
    #print(df)
    dirdict = {"Direction": ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W',
                             'WNW', 'NW', 'NNW']}

    fig = px.bar_polar(df, r='frequency', theta="Direction",
                       color="Interval (cm/s)", template="plotly_white",
                       color_discrete_sequence=px.colors.sequential.Plasma_r,
                       category_orders=dirdict,
                       title="Rosa de viento"
                       )
    fig.show()
