# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:37:57 2021
Alcance: Análisis de los datos referentes al campo de velocidades en el punto de fondeo de la plataforma W2P
"""


''' ##############################################  Carga de módulos #######################################'''
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

############################################### Fin de la Carga de Módulos ####################################

''' ################################### DEFINICIÓN DE FUNCIONES ###############################################'''
#__________________________________________ Inicio BoxPLot diario ________________________________________


def tsdf_d_boxplot(DataF,record=0,path=0):

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
            data_ = pd.concat([data_,aux], ignore_index=True, axis=1)
        
    data_.columns = name_
    #data = data.dropna(how="all", axis =1)  
    #data_.boxplot()
    fig = px.box(data_)
    fig.update_layout(title_text = "Diagrama de Cajas y Bigotes (por día) de la serie Temporal: " + DataF.name + " (" + netcdf_file_in_mo + "/2019)",
                      title_x=0.5,                      
                      font = dict (size = 18))
    fig.update_xaxes(title_text="Día del año")
    fig.update_yaxes(title_text="Dispersión")
    
    if record == 1:
        if not os.path.exists(path+DataF.name):
            os.makedirs(path+DataF.name)    
        fig.write_image(path+DataF.name+"/boxplot_diario.png", width=1980, height=1080) 
 
    return plot(fig)

#___________________________________________ Fin BoxPlot Diario _____________________________________________________

#__________________________________________ Inicio BoxPLot Semanas___________________________________________________


def tsdf_W_boxplot(DataF,record=0,path=0):

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
            data_ = pd.concat([data_,aux], ignore_index=True, axis=1)
        
    data_.columns = name_
    #data = data.dropna(how="all", axis =1)  
    #data_.boxplot()
    fig = px.box(data_)
    fig.update_layout(title_text = "Diagrama de Cajas y Bigotes (por semana) de la serie Temporal: " + DataF.name + " (" + netcdf_file_in_mo + "/2019)",
                      title_x=0.5,                      
                      font = dict (size = 18))
    fig.update_xaxes(title_text="Semana del año")
    fig.update_yaxes(title_text="Dispersión")
    
    if record == 1:
        if not os.path.exists(path+DataF.name):
            os.makedirs(path+DataF.name)    
        fig.write_image(path+DataF.name+"/boxplot_semanal.png", width=1980, height=1080) 
 
    return plot(fig)

#___________________________________________ Fin BoxPlot Semanas _____________________________________________________

#___________________________________________ Función de distribución e histograma ____________________________

def ts_density(DataF, bins_= 0.1,record=0,path=0):
    hist_data = DataF.dropna()
    hist_data = [hist_data]
    #hist_data = [[1,1,1,3,3,5,5]]
    group_labels = [DataF.name]
    fig2 = ff.create_distplot(hist_data, group_labels,bin_size=bins_, colors=['#66CDAA'], histnorm="probability")
    fig2.update_layout(title_text = "Distribución de la serie Temporal: " + DataF.name + " (" + netcdf_file_in_mo + "/2019)",
                      title_x=0.5,                      
                      font = dict (size = 18))
    if record == 1:
        if not os.path.exists(path+DataF.name):
            os.makedirs(path+DataF.name)    
        fig2.write_image(path+DataF.name+"/densidad.png", width=1980, height=1080) 
        
    return plot(fig2)


#___________________________________________ Fin de Función de Distribución e Histograma _____________________
        
#__________________________________________ Descomposición aditiva de Series temporales  _____________

def serie_desc(DataF,period_=60,record=0,path=0): #Por defecto el periodo es 60, ya que la frecuencia con la que trabajo es minuto. Uso intervalos de 60 minutos
    
    data = DataF.dropna()
    
    # Descomposición aditiva de las serie
    desc = sm.tsa.seasonal_decompose(data,model="aditive",period=period_)
    
    fig = make_subplots(rows = 4, cols = 1)
    
    fig.append_trace(go.Scatter(
        x= desc.observed.index,
        y= desc.observed.values,
        #line_color='cadetblue',
        marker = dict(size=7, color="yellowgreen", line = dict(width=.5, color='slategray')),
        line = dict(color="darkgreen"),
        mode='lines+markers',
        name = DataF.name,
        ), row=1, col=1)

    fig.append_trace(go.Scatter(
        x= desc.trend.index,
        y= desc.trend.values,
        line_color='indianred',
        name = "Tendencia",
        ), row=2, col=1)

    fig.append_trace(go.Scatter(
        x= desc.seasonal.index,
        y= desc.seasonal.values,
        line_color='slategrey',
        name = "Estacionalidad",
        ), row=3, col=1)
    
    fig.append_trace(go.Scatter(
        x= desc.resid.index,
        y= desc.resid.values,
        line_color='olive',
        name = "Aleatoriedad",
        ), row=4, col=1)
    
    fig.update_layout(title_text = "Descomposición de la serie Temporal: " + DataF.name + ".    " + "Periodo seleccionado: " + str(period_),
                      title_x=0.5,
                      font = dict (size = 20))
    if record == 1:
        if not os.path.exists(path+DataF.name):
            os.makedirs(path+DataF.name)    
        fig.write_image(path+DataF.name+"/desc"+str(period_)+".png", width=1980, height=1080)   
            
    return plot(fig)
    
#__________________________________________ Fin de Descomposición aditiva de Series temporales  _____________

# __________________________________________ Representación de corrientes promedios por hora del mes  _____________

def histo_hora(DataF,record=0,path=0):
    hist_data = DataF.dropna()

    fig = make_subplots(rows=2, cols=1)
    #fig = go.Figure()
    # magnitud promedio
    # mag_prom = (hist_data['v'] ** 2 + hist_data['u'] ** 2) ** 0.5

    df_hist_data = pd.DataFrame(hist_data)
    df_hist_data.reset_index(inplace=True)
    df_hist_data['Hours'] = df_hist_data['time'].dt.hour

    fig.append_trace(go.Box(x=df_hist_data['Hours'], y=df_hist_data['modulo'], name = "Velocidad"), row=1, col=1)
    fig.update_layout(title_text="Magnitud de corriente por Hora: " + DataF.name + " (" + netcdf_file_in_mo + "/2019)",
                      title_x=0.5,
                      font=dict(size=18),
                      yaxis_title_text='Velocidad de corriente (m/s)',  # xaxis label
                      xaxis_title_text='Hora',  # yaxis label
                      bargap=0.0  # gap between bars of the same location coordinates
                      )

    df_hist_data = df_hist_data.groupby(df_hist_data['Hours']).mean().reset_index()
    fig.append_trace(go.Bar(x = df_hist_data['Hours'], y = df_hist_data['modulo'], name = "Velocidad media"),
                     row=2, col=1)

    if record == 1:
        if not os.path.exists(path + DataF.name):
            os.makedirs(path + DataF.name)
        fig.write_image(path + DataF.name + "/horas.png", width=1980, height=1080)

    return plot(fig)


# __________________________________________ Fin de Representación de corrientes promedios por hora del mes  _____________

######################################## FIN DE DEFINCIÓN DE FUNCIONES #############################################

'''######################################### CARGA Y PREPARACIÓN DE LOS DATOS #################################### '''

''' Cargamos el fichero con los datos '''
netcdf_file_in = './W2P/Data/Plocan_201906.nc'
netcdf_file_in_mo = netcdf_file_in[22:24] # Digitos de mes, seran usados para almacenar imagenes
ds = xr.open_dataset(netcdf_file_in)

''' Manipulamos los datos para convertirlos en serie temporal '''
# Pasamos la información en netcedf a un data frame
w2p_3ind = ds.to_dataframe() # Vemos que tiene tres índices: Longitud, latitud y tiempo

# Vamos a convertir ahora en una serie temporal donde el index es el tiempo y el resto variables
w2p_2ind = w2p_3ind.reset_index(level='lon')
w2p_1ind = w2p_2ind.reset_index(level='lat')

# Según sabemos, la plataforma estuvo FONDEADA en las siguientes coordendas UTM WGS84 Huso 28N,
# x = 464230.1
# y = 3100543.9
# Esto, convertido a longitud y latitud, tenemos que:
Latitud = 28.029683035016497 #Latitud de Fondeo
Longitud = -15.363900078354046 #Longitud de Fondeo

################################# FIN DE CARGA Y PREPARACIÓN DE LOS DATOS ########################################


'''############################### Ubicación del punto de fondeo y definición del área de análisis ##############'''

# Dato más cercano al punto de Fondeo.

coord_lat = abs(w2p_1ind.lat.unique()-Latitud) # Sacamos la diferencia entre la latitud deseada y las disponibles
coord_lat = pd.DataFrame(coord_lat).set_index(w2p_1ind.lat.unique()) # Ponemos las latitudes com índices 
coord_lat = coord_lat[0].idxmin() # Sacamos el índice del menor valor (más cercano)   

coord_lon = abs(w2p_1ind.lon.unique()-Longitud) # Sacamos la diferencia entre la latitud deseada y las disponibles
coord_lon = pd.DataFrame(coord_lon).set_index(w2p_1ind.lon.unique()) # Ponemos las latitudes com índices 
coord_lon = coord_lon[0].idxmin() # Sacamos el índice del menor valor (más cercano) 

# Definción del área de análisis

latitudes = w2p_1ind.lat.unique()
a_lat = list(map(str,latitudes)) # Lista de latitudes
longitudes = w2p_1ind.lon.unique()
a_lon = list(map(str,longitudes)) # Lisa de Longitudes


ind_lat = np.where(latitudes == coord_lat)[0][0] # Índice de la latitud más cercana al punto de referencia
ind_lon = np.where(longitudes == coord_lon)[0][0] #Índice de la longitud más cercana al punto de referencia

if coord_lat < Latitud: # Si la latitud del punto de fondeo es mayor que la del punto más cercano entonces:
    ind_lat_1 = ind_lat # La y1 de mi área estará en la latitud del punto más cercano (por debajo del punto de fondeo)
    ind_lat_2 = ind_lat+1 # La y2 de mi área estará en la latitud siguiente al punto cercano (por encima del punto de fondeo)  
else: # Si la lalitud del punto de fondeo es menor
    ind_lat_1 = ind_lat-1 # Entonces y1 tendrá que ser la inferior (para quedar por debajo)
    ind_lat_2 = ind_lat # y2 será la lalitud del punto más cercao (que está por encima)
if coord_lon < Longitud:
    ind_lon_1 = ind_lon
    ind_lon_2 = ind_lon+1
else:
    ind_lon_1 = ind_lon-1
    ind_lon_2 = ind_lon

## Representación de la cuadrícula de datos
x = [ind_lon_1, ind_lon_1, ind_lon_2, ind_lon_2,ind_lon_1]
y = [ind_lat_1, ind_lat_2, ind_lat_1, ind_lat_2,ind_lat_1]

fig, ax = plt.subplots(figsize=(10, 8))
plt.plot(x, y)

ax = plt.gca()

# Set axis ranges; by default this will put major ticks every 25.
ax.set_xlim(0, longitudes.shape[0])
ax.set_ylim(0, latitudes.shape[0])

# Change major ticks to show every 20.
ax.xaxis.set_major_locator(MultipleLocator(20))
ax.yaxis.set_major_locator(MultipleLocator(20))

# Change minor ticks to show every 5. (20/4 = 5)
ax.xaxis.set_minor_locator(AutoMinorLocator(4))
ax.yaxis.set_minor_locator(AutoMinorLocator(4))

# Turn grid on for both major and minor ticks and style minor slightly
# differently.
ax.grid(which='major', color='#CCCCCC', linestyle='--')
ax.grid(which='minor', color='#CCCCCC', linestyle=':')

plt.xticks(np.arange(longitudes.shape[0]),a_lon,rotation='vertical')
plt.yticks(np.arange(latitudes.shape[0]),a_lat,rotation='horizontal')
plt.xlabel("longitud")
plt.ylabel("Latitud")

############################### Fin de Ubicación del punto de fondeo y definición del área de análisis ######



'''#######  Análisis de las series temporales correspondientes a los 4 puntos que definen el área ################## '''

# Variables para guardar datos
path = ("./W2P/Gráficas/" + netcdf_file_in_mo + "/") #Guardado de las gráficas
grabar = 0 # Para guardar o no las gráficas (1 graba/0 no graba)


# Obtención de los cuatro puntos
dato_SW = w2p_1ind[w2p_1ind.lat==latitudes[ind_lat_1]]
dato_SW = dato_SW[dato_SW.lon==longitudes[ind_lon_1]]
dato_SW["modulo"] = np.sqrt(dato_SW.u * dato_SW.u + dato_SW.v * dato_SW.v)
dato_SW["dir"] = ((360*np.arctan2(dato_SW.u,dato_SW.v)/(2*np.pi)) + 360) % 360
dato_SW.iloc[:,[0,1,8,9]].describe()

dato_SE = w2p_1ind[w2p_1ind.lat==latitudes[ind_lat_1]]
dato_SE = dato_SE[dato_SE.lon==longitudes[ind_lon_2]]
dato_SE["modulo"] = np.sqrt(dato_SE.u * dato_SE.u + dato_SE.v * dato_SE.v)
dato_SE["dir"] = ((360*np.arctan2(dato_SE.u,dato_SE.v)/(2*np.pi)) + 360) % 360
dato_SE.iloc[:,[0,1,8,9]].describe()

dato_NW = w2p_1ind[w2p_1ind.lat==latitudes[ind_lat_2]]
dato_NW = dato_NW[dato_NW.lon==longitudes[ind_lon_1]]
dato_NW["modulo"] = np.sqrt(dato_NW.u * dato_NW.u + dato_NW.v * dato_NW.v)
dato_NW["dir"] = ((360*np.arctan2(dato_NW.u,dato_NW.v)/(2*np.pi)) + 360) % 360
dato_NW.iloc[:,[0,1,8,9]].describe()

dato_NE = w2p_1ind[w2p_1ind.lat==latitudes[ind_lat_2]]
dato_NE = dato_NE[dato_NE.lon==longitudes[ind_lon_2]]
dato_NE["modulo"] = np.sqrt(dato_NE.u * dato_NE.u + dato_NE.v * dato_NE.v)
dato_NE["dir"] = ((360*np.arctan2(dato_NE.u,dato_NE.v)/(2*np.pi)) + 360) % 360
dato_NE.iloc[:,[0,1,8,9]].describe()

dato_lejos = w2p_1ind[w2p_1ind.lat==latitudes[50]] #Para ver si en un lugar alejado del punto hay comportamiento similar
dato_lejos = dato_lejos[dato_lejos.lon==longitudes[ind_lon_1]]
dato_lejos["modulo"] = np.sqrt(dato_lejos.u * dato_lejos.u + dato_lejos.v * dato_lejos.v)
dato_lejos["dir"] = ((360*np.arctan2(dato_lejos.u,dato_lejos.v)/(2*np.pi)) + 360) % 360
dato_lejos.iloc[:,[0,1,8,9]].describe()

# Correlaciones para todos los meses
#
# data_to_test = 'dir'
# corrdf = pd.DataFrame()
# corrdf['NW'] = dato_NW[data_to_test]
# corrdf['NE'] = dato_NE[data_to_test]
# corrdf['SW'] = dato_SW[data_to_test]
# corrdf['SE'] = dato_SE[data_to_test]
#
# corr = corrdf.corr()
# print(corr)
#
# # Correlaciones entre modulo y direccion
#
# corr2 = pd.DataFrame()
# corr2['NW_dir'] = dato_NW['dir']
# corr2['NW_modulo'] = dato_NW['modulo']
# corr2['NE_dir'] = dato_NE['dir']
# corr2['NE_modulo'] = dato_NE['modulo']
#
# corr2 = corr2.corr()
# print(corr2)
#
# corr2 = pd.DataFrame()
# corr2['SW_dir'] = dato_SW['dir']
# corr2['SW_modulo'] =dato_SW['modulo']
# corr2['SE_dir'] = dato_SE['dir']
# corr2['SE_modulo'] = dato_SE['modulo']
# corr2 = corr2.corr()
# print(corr2)

# Representación de las series temporales: Dirección

direcciones =  go.Figure()

direcciones.add_trace(go.Scatter(x=dato_SW.index, y=dato_SW.dir,
                    marker = dict(size=7, color="plum", line = dict(width=.5, color='slategray')),
                    line = dict(color="purple"),
                    mode='lines+markers',
                    name='dirección_SW'))

direcciones.add_trace(go.Scatter(x=dato_SE.index, y=dato_SE.dir,
                    marker = dict(size=7, color="yellowgreen", line = dict(width=.5, color='slategray')),
                    line = dict(color="darkgreen"),
                    mode='lines+markers',
                    name='dirección_SE'))

direcciones.add_trace(go.Scatter(x=dato_NW.index, y=dato_NW.dir,
                    marker = dict(size=7, color="lightsalmon", line = dict(width=.5, color='slategray')),
                    line = dict(color="tomato"),
                    mode='lines+markers',
                    name='dirección_NW'))


direcciones.add_trace(go.Scatter(x=dato_lejos.index, y=dato_lejos.dir,
                    marker = dict(size=7, color="orange", line = dict(width=.5, color='slategray')),
                    line = dict(color="gold"),
                    mode='lines+markers',
                    name='dirección_NE'))

direcciones.update_yaxes(title_text="Dirección de la velocidad en grados (0º Norte)")
direcciones.update_xaxes(title_text="Tiempo")
direcciones.update_layout(title_text = "Dirección de la velocidad para los 4 puntos estudiados (" + netcdf_file_in_mo + "/2019)",
                      title_x=0.5,
                      font = dict (size = 20))

plot(direcciones)

if not os.path.exists(path+"general"): # Comprobamos si existe el directorio y sino lo creamos
    os.makedirs(path+"general")

if grabar == 1:
    direcciones.write_image(path+"general/"+"direcciones.png", width=1980, height=1080)     
          
# Representación de las series temporales: Módulo

modulos =  go.Figure()

modulos.add_trace(go.Scatter(x=dato_SW.index, y=dato_SW.modulo,
                    marker = dict(size=7, color="plum", line = dict(width=.5, color='slategray')),
                    line = dict(color="purple"),
                    mode='lines+markers',
                    name='modulo_SW'))

modulos.add_trace(go.Scatter(x=dato_SE.index, y=dato_SE.modulo,
                    marker = dict(size=7, color="yellowgreen", line = dict(width=.5, color='slategray')),
                    line = dict(color="darkgreen"),
                    mode='lines+markers',
                    name='modulo_SE'))

modulos.add_trace(go.Scatter(x=dato_NW.index, y=dato_NW.modulo,
                    marker = dict(size=7, color="lightsalmon", line = dict(width=.5, color='slategray')),
                    line = dict(color="tomato"),
                    mode='lines+markers',
                    name='modulo_NW'))


modulos.add_trace(go.Scatter(x=dato_NE.index, y=dato_NE.modulo,
                    marker = dict(size=7, color="orange", line = dict(width=.5, color='slategray')),
                    line = dict(color="gold"),
                    mode='lines+markers',
                    name='modulo_NE'))

modulos.update_yaxes(title_text="Módulo de la velocidad en m/s")
modulos.update_xaxes(title_text="Tiempo")
modulos.update_layout(title_text = "Módulo de la velocidad para los 4 puntos estudiados (" + netcdf_file_in_mo + "/2019)",
                      title_x=0.5,
                      font = dict (size = 20))

plot(modulos)
if not os.path.exists(path+"general"): # Comprobamos si existe el directorio y sino lo creamos
    os.makedirs(path+"general")

if grabar == 1:
    modulos.write_image(path+"general"+"/modulos.png", width=1980, height=1080) 

# Punto sobre el que se va llevar el análisis:
    
dato = dato_NW # Serie temporal que voy a analizar
n_dato = "NW"  # Etiqueta para referenciar después las gráficas a esos datos

# Representamos velocidad y dirección en el mismo gráfico para ver visualmente relación. Ejemplo para un punto

mod_dir =  make_subplots(specs=[[{"secondary_y": True}]]) # Vamos a poner dos ejes Y
mod_dir.add_trace(go.Scatter(x=dato.index, y=dato.modulo,
                    marker = dict(size=7, color="plum", line = dict(width=.5, color='slategray')),
                    line = dict(color="purple"),
                    mode='lines+markers',
                    name='Modulo'),
                    secondary_y = False)

mod_dir.add_trace(go.Scatter(x=dato.index, y=dato.dir,
                    marker = dict(size=7, color="yellowgreen", line = dict(width=.5, color='slategray')),
                    line = dict(color="darkgreen"),
                    mode='lines+markers',
                    name='Dirección'),
                    secondary_y = True)

# Títulos de los ejes Y
mod_dir.update_yaxes(title_text="Velocidad de la Corriente m/s", secondary_y=False)
mod_dir.update_yaxes(title_text="Orientación de la Corriente (Norte 0º. Sentido Horario)", secondary_y=True)
mod_dir.update_xaxes(title_text="Tiempo")
mod_dir.update_layout(title_text = "Módulo y Dirección de la Corriente en el punto " + n_dato + " (" + netcdf_file_in_mo + "/2019)",
                      title_x=0.5,
                      font = dict (size = 20))


plot(mod_dir)
if not os.path.exists(path+n_dato): # Comprobamos si existe el directorio y sino lo creamos
    os.makedirs(path+n_dato)

if grabar == 1:
    mod_dir.write_image(path+n_dato+"/mod_dir_"+n_dato+".png", width=1980, height=1080)

## Correlaciones entre dos puntos (modulo o dirección):
dato_NW.modulo.corr(dato_NE.modulo)

## Rosa de Corrientes
fig, ax = plt.subplots(figsize=(10, 8))
plt.plot()
ax = plt.gca()
ax = WindroseAxes.from_ax()
ax.bar(dato.dir, dato.modulo, normed=True, opening=0.8, edgecolor='white')
#ax.set_xticklabels(['N', 'NE',  'E', 'SE', 'S', 'SW','W', 'NW'])
#ax.set_xticklabels(['E', 'ENE', "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"])
ax.set_xticklabels(['E', "NE", "N", "NW", "W", "SW", "S", "SE"])
#ax.set_xticklabels(["N", "NW", "W", "SW", "S", "SE",'E', "NE", ])
#ax.set_theta_zero_location('E') #Importante para darle sentido de dirección al gráfico
ax.set_legend()
plt.title("Rosa de Corrientes para el punto "+n_dato+" (" + netcdf_file_in_mo + "/2019)")
#plt.show()


# Hay que especificar si es el módulo o la dirección lo que vamos a analizar

dato = dato.modulo
dftest_NaNd = dato.dropna() # Dato sin valores nulos

''' 1. Bigotes por semana y días '''
tsdf_d_boxplot(dato,grabar,path)
tsdf_W_boxplot(dato,grabar,path)

''' 2. Dispersión e histograma '''
ts_density(dato,0.02,grabar,path) #0.02 para módulos, 24 para direcciones

''' 3. Descomposición aditiva de la serie'''
serie_desc(dato,24,grabar,path)

''' 4. Estacionariedad '''
print ('Result of Dickey-Fuller Test')
dftest = adfuller(dftest_NaNd, autolag= "AIC")
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', '#Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

print ('Results of KPSS Test:') # Parece que esta forma de medir la estacionariedad es mejor
kpsstest = kpss(dftest_NaNd,nlags="auto")
kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
for key,value in kpsstest[3].items():
    kpss_output['Critical Value (%s)'%key] = value
print (kpss_output)

''' 5. Análisis de la estacionalidad '''
plot_acf(dftest_NaNd, lags = 4*24, zero = False) # 4 muestras de 24 puntos: 4 días
plot_pacf(dftest_NaNd, lags = 4*24, zero = False, method = ("ols"))
pd.plotting.lag_plot(dftest_NaNd)

''' 6. Histograma magnitud por horas '''
histo_hora(dato,grabar,path) #0.02 para módulos, 24 para direcciones

######## Fin de Análisis de las series temporales correspondientes a los 4 puntos que definen el área #########


''''######## Variado de cosas que he ido probando para ver otras posibles representciones ###############'''


### Ejemplo por si quisiéramos hacer el histograma por horas:
# dato_NE_hora = dato_NE
# dato_NE_hora = dato_NE.groupby(dato_NE.index.hour).mean()
#
# ax = dato_NE_hora["modulo"].plot(kind='bar', color='b')
# ax = dato_NE_hora["dir"].plot(kind='bar', color='b')
#
#
# #### Gráfico de representación de patrones
#
# dato_pat = dato_NE #
#
# ## Representamos esto matplotlib:
#
# dato_pat["date"] = dato_pat.index.normalize()
# dato_pat["date"] = dato_pat['date'].dt.date
# # dato_pat["date"] = dato_pat['date'].dt.date.astype(str)
# dato_pat["dtime"] = dato_pat.index.time
# dato_pat = dato_pat.pivot(index="date",columns="dtime",values="dir") # Con esto tenemos los datos por horas para cada día
#
#
# fig, ax = plt.subplots(1, 1, figsize = (10,6))
# dato_pat.T.plot(ax=ax, color= "C0", alpha=0.5, legend = False)
# ax.set_xlim([0,24])
# ax.set_ylim(0,360)
# ax.set_ylabel("m/s")
# ax.set_title("Modulos en el punto NE")
#
# ## Representamos esto plotly:
#
#
# fig = px.line(dato_pat, x = "date", y="dir", color = "dtime") # Patrón por horas
# fig = px.line(dato_pat, x = "dtime", y="dir", color = "date") # Patrón pode días
# plot(fig)

######## Fin de Variado de cosas que he hido probando para ver otras posibles representciones ###############