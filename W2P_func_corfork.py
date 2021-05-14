## Imports
import xarray as xr
import os
import pandas as pd
from windrose import WindroseAxes
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import plotly.figure_factory as ff
import numpy as np
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
import plotly.figure_factory as ff
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.filters.hp_filter import hpfilter
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import glob as gb
from tqdm import tqdm
import matplotlib as mpl

mpl.rc('figure', max_open_warning = 0)

## Functions

def get_names(path):
    onlyfiles = gb.glob(path)
    return onlyfiles


def tsdf_d_boxplot(DataF, record=0, path=0, mo="", point="NE"):
    groups = DataF.dropna().groupby(pd.Grouper(freq="D"))

    name_ = []
    for name, group in groups:
        name_.append(str(name.month_name()) + "-" + str(name.day))

    data_ = pd.DataFrame()
    flag = 0

    for name, group in groups:
        if flag == 0:
            data_[str(name.month_name) + "-" + str(name.day)] = group.values
            flag = 1
        else:
            aux = pd.DataFrame(group.values)
            data_ = pd.concat([data_, aux], ignore_index=True, axis=1)

    data_.columns = name_
    # data = data.dropna(how="all", axis =1)
    # data_.boxplot()
    fig = px.box(data_)
    fig.update_layout(
        title_text="Diagrama de Cajas y Bigotes (por día) de la serie Temporal: " + DataF.name + " (" + mo + "/2019)",
        title_x=0.5,
        font=dict(size=18))
    fig.update_xaxes(title_text="Día del año")
    fig.update_yaxes(title_text="Dispersión")

    if record == 1:
        if not os.path.exists(path + point + "/" + DataF.name):
            os.makedirs(path + point + "/" + DataF.name)
        fig.write_image(path + point + "/" + DataF.name + "/boxplot_diario.png", width=1980, height=1080)

    #return plot(fig)


def tsdf_W_boxplot(DataF, record=0, path=0, mo="", point="NE"):
    groups = DataF.dropna().groupby(pd.Grouper(freq="W"))
    name_ = []
    for name, group in groups:
        name_.append(name.week)

    data_ = pd.DataFrame()
    flag = 0

    for name, group in groups:
        if flag == 0:
            data_[name.week] = group.values
            flag = 1
        else:
            aux = pd.DataFrame(group.values)
            data_ = pd.concat([data_, aux], ignore_index=True, axis=1)

    data_.columns = name_
    # data = data.dropna(how="all", axis =1)
    # data_.boxplot()
    fig = px.box(data_)
    fig.update_layout(
        title_text="Diagrama de Cajas y Bigotes (por semana) de la serie Temporal: " + DataF.name + " (" + mo + "/2019)",
        title_x=0.5,
        font=dict(size=18))
    fig.update_xaxes(title_text="Semana del año")
    fig.update_yaxes(title_text="Dispersión")

    if record == 1:
        if not os.path.exists(path + point + "/" + DataF.name):
            os.makedirs(path + point + "/" + DataF.name)
        fig.write_image(path + point + "/" + DataF.name + "/boxplot_semanal.png", width=1980, height=1080)

    #return plot(fig)


def ts_density(DataF, bins_=0.1, record=0, path=0, mo="", point="NE"):
    hist_data = DataF.dropna()
    hist_data = [hist_data]
    # hist_data = [[1,1,1,3,3,5,5]]
    group_labels = [DataF.name]
    fig2 = ff.create_distplot(hist_data, group_labels, bin_size=bins_, colors=['#66CDAA'], histnorm="probability")
    fig2.update_layout(
        title_text="Distribución de la serie Temporal: " + DataF.name + " (" + mo + "/2019)",
        title_x=0.5,
        font=dict(size=18))
    if record == 1:
        if not os.path.exists(path + point + "/" + DataF.name):
            os.makedirs(path + point + "/" + DataF.name)
        fig2.write_image(path + point + "/" + DataF.name + "/densidad.png", width=1980, height=1080)

    # return plot(fig2)


def serie_desc(DataF, period_=60, record=0,
               path=0, point="NE", mo="06"):  # Por defecto el periodo es 60, ya que la frecuencia con la que trabajo es minuto. Uso intervalos de 60 minutos

    data = DataF.dropna()

    # Descomposición aditiva de las serie
    desc = sm.tsa.seasonal_decompose(data, model="aditive", period=period_)

    fig = make_subplots(rows=4, cols=1)

    fig.append_trace(go.Scatter(
        x=desc.observed.index,
        y=desc.observed.values,
        # line_color='cadetblue',
        marker=dict(size=7, color="yellowgreen", line=dict(width=.5, color='slategray')),
        line=dict(color="darkgreen"),
        mode='lines+markers',
        name=DataF.name,
    ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x=desc.trend.index,
        y=desc.trend.values,
        line_color='indianred',
        name="Tendencia",
    ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x=desc.seasonal.index,
        y=desc.seasonal.values,
        line_color='slategrey',
        name="Estacionalidad",
    ), row=3, col=1)

    fig.append_trace(go.Scatter(
        x=desc.resid.index,
        y=desc.resid.values,
        line_color='olive',
        name="Aleatoriedad",
    ), row=4, col=1)

    fig.update_layout(
        title_text="Descomposición de la serie Temporal: " + DataF.name + ", " + point + " Periodo seleccionado: " + str(
            period_) + " (" + mo + "/2019)" ,
        title_x=0.5,
        font=dict(size=20))
    if record == 1:
        if not os.path.exists(path + point + "/" + DataF.name):
            os.makedirs(path + point + "/" + DataF.name)
        fig.write_image(path + point + "/" + DataF.name + "/desc" + str(period_) + ".png", width=1980, height=1080)

    #return plot(fig)

def histo_hora(DataF,record=0,path=0, point="NE", mo = ""):
    hist_data = DataF.dropna()

    fig = make_subplots(rows=2, cols=1)
    #fig = go.Figure()
    # magnitud promedio
    # mag_prom = (hist_data['v'] ** 2 + hist_data['u'] ** 2) ** 0.5

    df_hist_data = pd.DataFrame(hist_data)
    df_hist_data.reset_index(inplace=True)
    df_hist_data['Hours'] = df_hist_data['time'].dt.hour

    if 'modulo' in df_hist_data:
        yent = df_hist_data['modulo']
    else:
        yent = df_hist_data['dir']

    fig.append_trace(go.Box(x=df_hist_data['Hours'], y=yent, name = "Velocidad"), row=1, col=1)
    fig.update_layout(title_text="Magnitud de corriente por Hora: " + DataF.name + " (" + mo + "/2019)",
                      title_x=0.5,
                      font=dict(size=18),
                      yaxis_title_text='Velocidad de corriente (m/s)',  # xaxis label
                      xaxis_title_text='Hora',  # yaxis label
                      bargap=0.0  # gap between bars of the same location coordinates
                      )

    df_hist_data = df_hist_data.groupby(df_hist_data['Hours']).mean().reset_index()
    fig.append_trace(go.Bar(x = df_hist_data['Hours'], y = yent, name = "Velocidad media"),
                     row=2, col=1)

    if record == 1:
        if not os.path.exists(path + point + "/" + DataF.name):
            os.makedirs(path + point + "/" + DataF.name)
        fig.write_image(path + point + "/" + DataF.name + "/horas.png", width=1980, height=1080)

    #return plot(fig)


def num2range(d, index=False):
    ranges = ['< 3.0', '3.0 - 5.0', '5.0 - 7.0', '7.0 - 10.0', '10.0 - 15.0', '15.0 - 20.0', '20.0 - 25.0', '25.0 - 30.0',
              '30.0 >']
    if d < 3:
        lx = 0
    elif d < 5:
        lx = 1
    elif d < 7:
        lx = 2
    elif d < 10:
        lx = 3
    elif d < 15:
        lx = 4
    elif d < 20:
        lx = 5
    elif d < 25:
        lx = 6
    elif d < 30:
        lx = 7
    else:
        lx = 8

    if index:
        return lx
    else:
        return ranges[lx]


def num2dir(d,deruta16=True):
    if deruta16: dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    else: dirs = ['N', 'NNE', 'ENE', 'E', 'ESE', 'SSE', 'S', 'SSW', 'WSW', 'W', 'WNW', 'NNW']
    ix = round(d / (360. / len(dirs)))
    return dirs[ix % len(dirs)]


def wrose(DataF, record=0, path=0, point="NE", mo = "",dfname=""):
    DataF = DataF.filter(['u','v'],axis=1)
    df = DataF.dropna()

    vmagn = (df['v'] ** 2 + df['u'] ** 2) ** 0.5
    vmagn = vmagn*100

    angle = np.arctan2(df['u'], df['v'])
    angle = np.degrees(angle)

    dfnew = {'Direction': angle, 'Interval (cm/s)': vmagn}
    df = pd.DataFrame(data=dfnew)

    df['Indexes'] = df['Interval (cm/s)'].apply(lambda x: num2range(x, index=True))
    df['Interval (cm/s)'] = df['Interval (cm/s)'].apply(lambda x: num2range(x))
    df['Direction'] = df['Direction'].apply(lambda x: num2dir(x))

    grp = df.groupby(["Indexes", "Direction", "Interval (cm/s)"]).size().reset_index(name="frequency")
    grp.sort_values(by='Indexes')

    dirdict = {"Direction": ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W',
                             'WNW', 'NW', 'NNW']}

    fig = px.bar_polar(grp, r='frequency', theta="Direction",
                       color="Interval (cm/s)", template="plotly_white",
                       color_discrete_sequence=px.colors.sequential.Plasma_r,
                       category_orders=dirdict,
                       title="Rosa de viento: " + dfname + ", " + point + " (" + mo + "/2019)"
                       )


    if record == 1:
        if not os.path.exists(path + point):
            os.makedirs(path + point)
        fig.write_image(path + point + "/rosa.png", width=500, height=500)

    #fig.show()


def w2p(netcdf_file_in, prints=False, modulo_entrada=True, bigdf=False, grabarval=1):

    if bigdf:
        netcdf_file_in_mo = "06-10"  # Digitos de mes, seran usados para almacenar imagenes
        w2p_3ind = big_df(netcdf_file_in)
    else:
        netcdf_file_in_mo = netcdf_file_in[22:24]  # Digitos de mes, seran usados para almacenar imagenes
        ds = xr.open_dataset(netcdf_file_in)
        w2p_3ind = ds.to_dataframe()

    # ''' Manipulamos los datos para convertirlos en serie temporal '''
    # # Pasamos la información en netcedf a un data frame
    # w2p_3ind = ds.to_dataframe()  # Vemos que tiene tres índices: Longitud, latitud y tiempo

    # Vamos a convertir ahora en una serie temporal donde el index es el tiempo y el resto variables
    w2p_2ind = w2p_3ind.reset_index(level='lon')
    w2p_1ind = w2p_2ind.reset_index(level='lat')

    # Según sabemos, la plataforma estuvo FONDEADA en las siguientes coordendas UTM WGS84 Huso 28N,
    # x = 464230.1
    # y = 3100543.9
    # Esto, convertido a longitud y latitud, tenemos que:
    Latitud = 28.029683035016497  # Latitud de Fondeo
    Longitud = -15.363900078354046  # Longitud de Fondeo

    ################################# FIN DE CARGA Y PREPARACIÓN DE LOS DATOS ########################################

    '''############################### Ubicación del punto de fondeo y definición del área de análisis ##############'''

    # Dato más cercano al punto de Fondeo.

    coord_lat = abs(w2p_1ind.lat.unique() - Latitud)  # Sacamos la diferencia entre la latitud deseada y las disponibles
    coord_lat = pd.DataFrame(coord_lat).set_index(w2p_1ind.lat.unique())  # Ponemos las latitudes com índices
    coord_lat = coord_lat[0].idxmin()  # Sacamos el índice del menor valor (más cercano)

    coord_lon = abs(
        w2p_1ind.lon.unique() - Longitud)  # Sacamos la diferencia entre la latitud deseada y las disponibles
    coord_lon = pd.DataFrame(coord_lon).set_index(w2p_1ind.lon.unique())  # Ponemos las latitudes com índices
    coord_lon = coord_lon[0].idxmin()  # Sacamos el índice del menor valor (más cercano)

    # Definción del área de análisis

    latitudes = w2p_1ind.lat.unique()
    a_lat = list(map(str, latitudes))  # Lista de latitudes
    longitudes = w2p_1ind.lon.unique()
    a_lon = list(map(str, longitudes))  # Lisa de Longitudes

    ind_lat = np.where(latitudes == coord_lat)[0][0]  # Índice de la latitud más cercana al punto de referencia
    ind_lon = np.where(longitudes == coord_lon)[0][0]  # Índice de la longitud más cercana al punto de referencia

    if coord_lat < Latitud:  # Si la latitud del punto de fondeo es mayor que la del punto más cercano entonces:
        ind_lat_1 = ind_lat  # La y1 de mi área estará en la latitud del punto más cercano (por debajo del punto de fondeo)
        ind_lat_2 = ind_lat + 1  # La y2 de mi área estará en la latitud siguiente al punto cercano (por encima del punto de fondeo)
    else:  # Si la lalitud del punto de fondeo es menor
        ind_lat_1 = ind_lat - 1  # Entonces y1 tendrá que ser la inferior (para quedar por debajo)
        ind_lat_2 = ind_lat  # y2 será la lalitud del punto más cercao (que está por encima)
    if coord_lon < Longitud:
        ind_lon_1 = ind_lon
        ind_lon_2 = ind_lon + 1
    else:
        ind_lon_1 = ind_lon - 1
        ind_lon_2 = ind_lon
    #
    # ## Representación de la cuadrícula de datos
    # x = [ind_lon_1, ind_lon_1, ind_lon_2, ind_lon_2, ind_lon_1]
    # y = [ind_lat_1, ind_lat_2, ind_lat_1, ind_lat_2, ind_lat_1]
    #
    # fig, ax = plt.subplots(figsize=(10, 8))
    # #plt.plot(x, y)
    #
    # ax = plt.gca()
    #
    # # Set axis ranges; by default this will put major ticks every 25.
    # ax.set_xlim(0, longitudes.shape[0])
    # ax.set_ylim(0, latitudes.shape[0])
    #
    # # Change major ticks to show every 20.
    # ax.xaxis.set_major_locator(MultipleLocator(20))
    # ax.yaxis.set_major_locator(MultipleLocator(20))
    #
    # # Change minor ticks to show every 5. (20/4 = 5)
    # ax.xaxis.set_minor_locator(AutoMinorLocator(4))
    # ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    #
    # # Turn grid on for both major and minor ticks and style minor slightly
    # # differently.
    # ax.grid(which='major', color='#CCCCCC', linestyle='--')
    # ax.grid(which='minor', color='#CCCCCC', linestyle=':')
    #
    # plt.xticks(np.arange(longitudes.shape[0]), a_lon, rotation='vertical')
    # plt.yticks(np.arange(latitudes.shape[0]), a_lat, rotation='horizontal')
    # plt.xlabel("longitud")
    # plt.ylabel("Latitud")

    ############################### Fin de Ubicación del punto de fondeo y definición del área de análisis ######

    '''#######  Análisis de las series temporales correspondientes a los 4 puntos que definen el área ################## '''

    # Variables para guardar datos
    path = ("./W2P/Gráficas/" + netcdf_file_in_mo + "/")  # Guardado de las gráficas
    grabar = grabarval  # Para guardar o no las gráficas (1 graba/0 no graba)

    datos = []

    # Obtención de los cuatro puntos
    dato_SW = w2p_1ind[w2p_1ind.lat == latitudes[ind_lat_1]]
    dato_SW = dato_SW[dato_SW.lon == longitudes[ind_lon_1]]
    dato_SW["modulo"] = np.sqrt(dato_SW.u * dato_SW.u + dato_SW.v * dato_SW.v)
    dato_SW["dir"] = ((360 * np.arctan2(dato_SW.u, dato_SW.v) / (2 * np.pi)) + 360) % 360
    dato_SW.iloc[:, [0, 1, 8, 9]].describe()

    datos.append(dato_SW)

    dato_SE = w2p_1ind[w2p_1ind.lat == latitudes[ind_lat_1]]
    dato_SE = dato_SE[dato_SE.lon == longitudes[ind_lon_2]]
    dato_SE["modulo"] = np.sqrt(dato_SE.u * dato_SE.u + dato_SE.v * dato_SE.v)
    dato_SE["dir"] = ((360 * np.arctan2(dato_SE.u, dato_SE.v) / (2 * np.pi)) + 360) % 360
    dato_SE.iloc[:, [0, 1, 8, 9]].describe()

    datos.append(dato_SE)

    dato_NW = w2p_1ind[w2p_1ind.lat == latitudes[ind_lat_2]]
    dato_NW = dato_NW[dato_NW.lon == longitudes[ind_lon_1]]
    dato_NW["modulo"] = np.sqrt(dato_NW.u * dato_NW.u + dato_NW.v * dato_NW.v)
    dato_NW["dir"] = ((360 * np.arctan2(dato_NW.u, dato_NW.v) / (2 * np.pi)) + 360) % 360
    dato_NW.iloc[:, [0, 1, 8, 9]].describe()

    datos.append(dato_NW)

    dato_NE = w2p_1ind[w2p_1ind.lat == latitudes[ind_lat_2]]
    dato_NE = dato_NE[dato_NE.lon == longitudes[ind_lon_2]]
    dato_NE["modulo"] = np.sqrt(dato_NE.u * dato_NE.u + dato_NE.v * dato_NE.v)
    dato_NE["dir"] = ((360 * np.arctan2(dato_NE.u, dato_NE.v) / (2 * np.pi)) + 360) % 360
    dato_NE.iloc[:, [0, 1, 8, 9]].describe()

    datos.append(dato_NE)

    dato_lejos = w2p_1ind[
        w2p_1ind.lat == latitudes[50]]  # Para ver si en un lugar alejado del punto hay comportamiento similar
    dato_lejos = dato_lejos[dato_lejos.lon == longitudes[ind_lon_1]]
    dato_lejos["modulo"] = np.sqrt(dato_lejos.u * dato_lejos.u + dato_lejos.v * dato_lejos.v)
    dato_lejos["dir"] = ((360 * np.arctan2(dato_lejos.u, dato_lejos.v) / (2 * np.pi)) + 360) % 360
    dato_lejos.iloc[:, [0, 1, 8, 9]].describe()

    # for dato in datos:
    #     print(dato.describe())

    # Correlaciones para todos los meses

    data_to_test = 'modulo'
    corrdf = pd.DataFrame()
    corrdf['NW'] = dato_NW[data_to_test]
    corrdf['NE'] = dato_NE[data_to_test]
    corrdf['SW'] = dato_SW[data_to_test]
    corrdf['SE'] = dato_SE[data_to_test]

    print('\n')
    corr = corrdf.corr()
    print(corr)

    # Correlaciones entre modulo y direccion

    corr2 = pd.DataFrame()
    corr2['NW_dir'] = dato_NW['dir']
    corr2['NW_modulo'] = dato_NW['modulo']
    corr2['NE_dir'] = dato_NE['dir']
    corr2['NE_modulo'] = dato_NE['modulo']

    corr2 = corr2.corr()
    print(corr2)

    corr2 = pd.DataFrame()
    corr2['SW_dir'] = dato_SW['dir']
    corr2['SW_modulo'] =dato_SW['modulo']
    corr2['SE_dir'] = dato_SE['dir']
    corr2['SE_modulo'] = dato_SE['modulo']
    corr2 = corr2.corr()
    print(corr2)

    # Representación de las series temporales: Dirección

    direcciones = go.Figure()

    direcciones.add_trace(go.Scatter(x=dato_SW.index, y=dato_SW.dir,
                                     marker=dict(size=7, color="plum", line=dict(width=.5, color='slategray')),
                                     line=dict(color="purple"),
                                     mode='lines+markers',
                                     name='dirección_SW'))

    direcciones.add_trace(go.Scatter(x=dato_SE.index, y=dato_SE.dir,
                                     marker=dict(size=7, color="yellowgreen", line=dict(width=.5, color='slategray')),
                                     line=dict(color="darkgreen"),
                                     mode='lines+markers',
                                     name='dirección_SE'))

    direcciones.add_trace(go.Scatter(x=dato_NW.index, y=dato_NW.dir,
                                     marker=dict(size=7, color="lightsalmon", line=dict(width=.5, color='slategray')),
                                     line=dict(color="tomato"),
                                     mode='lines+markers',
                                     name='dirección_NW'))

    direcciones.add_trace(go.Scatter(x=dato_lejos.index, y=dato_lejos.dir,
                                     marker=dict(size=7, color="orange", line=dict(width=.5, color='slategray')),
                                     line=dict(color="gold"),
                                     mode='lines+markers',
                                     name='dirección_NE'))

    direcciones.update_yaxes(title_text="Dirección de la velocidad en grados (0º Norte)")
    direcciones.update_xaxes(title_text="Tiempo")
    direcciones.update_layout(
        title_text="Dirección de la velocidad para los 4 puntos estudiados (" + netcdf_file_in_mo + "/2019)",
        title_x=0.5,
        font=dict(size=20))

    #plot(direcciones)

    if not os.path.exists(path + "general"):  # Comprobamos si existe el directorio y sino lo creamos
        os.makedirs(path + "general")

    if grabar == 1:
        direcciones.write_image(path + "general/" + "direcciones.png", width=1980, height=1080)

        # Representación de las series temporales: Módulo

    modulos = go.Figure()

    modulos.add_trace(go.Scatter(x=dato_SW.index, y=dato_SW.modulo,
                                 marker=dict(size=7, color="plum", line=dict(width=.5, color='slategray')),
                                 line=dict(color="purple"),
                                 mode='lines+markers',
                                 name='modulo_SW'))

    modulos.add_trace(go.Scatter(x=dato_SE.index, y=dato_SE.modulo,
                                 marker=dict(size=7, color="yellowgreen", line=dict(width=.5, color='slategray')),
                                 line=dict(color="darkgreen"),
                                 mode='lines+markers',
                                 name='modulo_SE'))

    modulos.add_trace(go.Scatter(x=dato_NW.index, y=dato_NW.modulo,
                                 marker=dict(size=7, color="lightsalmon", line=dict(width=.5, color='slategray')),
                                 line=dict(color="tomato"),
                                 mode='lines+markers',
                                 name='modulo_NW'))

    modulos.add_trace(go.Scatter(x=dato_NE.index, y=dato_NE.modulo,
                                 marker=dict(size=7, color="orange", line=dict(width=.5, color='slategray')),
                                 line=dict(color="gold"),
                                 mode='lines+markers',
                                 name='modulo_NE'))

    modulos.update_yaxes(title_text="Módulo de la velocidad en m/s")
    modulos.update_xaxes(title_text="Tiempo")
    modulos.update_layout(
        title_text="Módulo de la velocidad para los 4 puntos estudiados (" + netcdf_file_in_mo + "/2019)",
        title_x=0.5,
        font=dict(size=20))

    #plot(modulos)
    if not os.path.exists(path + "general"):  # Comprobamos si existe el directorio y sino lo creamos
        os.makedirs(path + "general")

    if grabar == 1:
        modulos.write_image(path + "general" + "/modulos.png", width=1980, height=1080)

        # Punto sobre el que se va llevar el análisis:

    #dato = dato_NW  # Serie temporal que voy a analizar
    n_dato_index = ["SW","SE","NW","NE"]  # Etiqueta para referenciar después las gráficas a esos datos
    cn_dato = 0

    # Representamos velocidad y dirección en el mismo gráfico para ver visualmente relación. Ejemplo para un punto

    for dato in datos:
        n_dato = n_dato_index[cn_dato]
        print(n_dato)
        mod_dir = make_subplots(specs=[[{"secondary_y": True}]])  # Vamos a poner dos ejes Y
        mod_dir.add_trace(go.Scatter(x=dato.index, y=dato.modulo,
                                     marker=dict(size=7, color="plum", line=dict(width=.5, color='slategray')),
                                     line=dict(color="purple"),
                                     mode='lines+markers',
                                     name='Modulo'),
                          secondary_y=False)

        mod_dir.add_trace(go.Scatter(x=dato.index, y=dato.dir,
                                     marker=dict(size=7, color="yellowgreen", line=dict(width=.5, color='slategray')),
                                     line=dict(color="darkgreen"),
                                     mode='lines+markers',
                                     name='Dirección'),
                          secondary_y=True)

        # Títulos de los ejes Y
        mod_dir.update_yaxes(title_text="Velocidad de la Corriente m/s", secondary_y=False)
        mod_dir.update_yaxes(title_text="Orientación de la Corriente (Norte 0º. Sentido Horario)", secondary_y=True)
        mod_dir.update_xaxes(title_text="Tiempo")
        mod_dir.update_layout(
            title_text="Módulo y Dirección de la Corriente en el punto " + n_dato + " (" + netcdf_file_in_mo + "/2019)",
            title_x=0.5,
            font=dict(size=20))

        #plot(mod_dir)
        if not os.path.exists(path + n_dato):  # Comprobamos si existe el directorio y sino lo creamos
            os.makedirs(path + n_dato)

        if grabar == 1:
            mod_dir.write_image(path + n_dato + "/mod_dir_" + n_dato + ".png", width=1980, height=1080)

        # Hay que especificar si es el módulo o la dirección lo que vamos a analizar

        datoWR = dato
        if modulo_entrada:
            dato = dato.modulo
        else:
            dato = dato.dir
        dftest_NaNd = dato.dropna()  # Dato sin valores nulos

        ''' Rosa de viento'''
        if modulo_entrada:
            wrose(datoWR, grabar, path, mo=netcdf_file_in_mo, point=n_dato, dfname=dato.name,)

        ''' 1. Bigotes por semana y días '''
        tsdf_d_boxplot(dato, grabar, path, mo=netcdf_file_in_mo, point=n_dato)
        tsdf_W_boxplot(dato, grabar, path, mo=netcdf_file_in_mo, point=n_dato)

        ''' 2. Dispersión e histograma '''
        if modulo_entrada:
            ts_density(dato, 0.02, grabar, path, mo=netcdf_file_in_mo, point=n_dato)  # 0.02 para módulos, 24 para direcciones
        else:
            ts_density(dato, 24, grabar, path, mo=netcdf_file_in_mo,
                       point=n_dato)  # 0.02 para módulos, 24 para direcciones

        ''' 3. Descomposición aditiva de la serie'''
        serie_desc(dato, 24, grabar, path, point=n_dato, mo=netcdf_file_in_mo)

        ''' 4. Estacionariedad '''
        if grabar:
            if not os.path.exists(path):
                os.makedirs(path)
            f = open(path + "/adfuller.txt", "w+")
            f.write('Result of Dickey-Fuller Test \n')
            dftest = adfuller(dftest_NaNd, autolag="AIC")
            dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', '#Observations Used'])
            for key, value in dftest[4].items():
                dfoutput['Critical Value (%s)' % key] = value
            f.write(str(dfoutput))
            f.close()

        if grabar:
            if not os.path.exists(path + "/"):
                os.makedirs(path + "/")
            f = open(path + "/kpss.txt", "w+")
            f.write('Results of KPSS Test: \n')  # Parece que esta forma de medir la estacionariedad es mejor
            kpsstest = kpss(dftest_NaNd, nlags="auto")
            kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
            for key, value in kpsstest[3].items():
                kpss_output['Critical Value (%s)' % key] = value
            f.write(str(kpss_output))
            f.close()

        ''' 5. Análisis de la estacionalidad '''
        plot_acf(dftest_NaNd, lags=4 * 24, zero=False)  # 4 muestras de 24 puntos: 4 días
        plot_pacf(dftest_NaNd, lags=4 * 24, zero=False, method=("ols"))

        #plt.show()
        pd.plotting.lag_plot(dftest_NaNd)
        #plt.show()

        ''' 6. Histograma magnitud por horas '''
        histo_hora(dato, grabar, path, mo=netcdf_file_in_mo, point=n_dato)  # 0.02 para módulos, 24 para direcciones

        cn_dato += 1

    ######## Fin de Análisis de las series temporales correspondientes a los 4 puntos que definen el área #########


def bigdf_simple(df_to_concat):
    e_df = pd.DataFrame()
    for x in df_to_concat:
        e_df.append(x)

    return e_df

def big_df(df_to_concat):
    e_df = pd.DataFrame()
    for x in df_to_concat:
        #print(x)
        ds = xr.open_dataset(x)
        df = ds.to_dataframe()
        #print(df)
        e_df = e_df.append(df)

    return e_df

## Main

path = './W2P/Data/'

plocan_names = get_names(path + '*.nc')
grabar = 0

with tqdm(total=2) as pbar:
    w2p(plocan_names, bigdf=True,grabarval=grabar)
    pbar.update(1)
    w2p(plocan_names,modulo_entrada=False, bigdf=True,grabarval=grabar)
    pbar.update(1)
print("Job done")