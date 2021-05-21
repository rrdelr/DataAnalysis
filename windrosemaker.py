from matplotlib import pyplot as plt
from windrose import WindroseAxes
from funcs import to_df, info, gen_scat, wrose

data = to_df("PARAMETROS_PROTOTIPO(WIP10+ 2019).xlsx")

## Rosa de Corrientes
fig, ax = plt.subplots(figsize=(10, 8))
#plt.plot()
ax = plt.gca()
ax = WindroseAxes.from_ax()
ax.bar(data[' WIND_DIRECTION'], data[' WIND_SPEED'], normed=True, opening=0.8, edgecolor='white')
#ax.set_xticklabels(['N', 'NE',  'E', 'SE', 'S', 'SW','W', 'NW'])
#ax.set_xticklabels(['E', 'ENE', "NE", "NNE", "N", "NNW", "NW", "WNW", "W", "WSW", "SW", "SSW", "S", "SSE", "SE", "ESE"])
ax.set_xticklabels(['E', "NE", "N", "NW", "W", "SW", "S", "SE"])
#ax.set_xticklabels(["N", "NW", "W", "SW", "S", "SE",'E', "NE", ])
#ax.set_theta_zero_location('E') #Importante para darle sentido de dirección al gráfico
ax.set_legend()
plt.title("Rosa de Corrientes")
plt.show()