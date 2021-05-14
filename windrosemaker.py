from matplotlib import pyplot as plt
from windrose import WindroseAxes
import xarray as xr
import numpy as np
import pandas as pd

netcdf_file_in = './W2P/Data/Plocan_201909.nc'
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


# Punto sobre el que se va llevar el análisis:

dato = dato_SE  # Serie temporal que voy a analizar
n_dato = "SE"  # Etiqueta para referenciar después las gráficas a esos datos

## Rosa de Corrientes
fig, ax = plt.subplots(figsize=(10, 8))
#plt.plot()
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
plt.show()