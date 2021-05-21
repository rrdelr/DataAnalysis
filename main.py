from funcs import to_df, info, gen_scat, wrose, sep_df, box_plot, avg_plot


# Main sequence
data = to_df("PARAMETROS_PROTOTIPO(WIP10+ 2019).xlsx")
info(data, ' WIND_SPEED', ' ROTOR_SPEED', ' V_OUT')
gen_scat(data, '06', '11', 'Date', ' ROTOR_SPEED', ' WIND_SPEED', ' V_OUT')
#
# # gen_vs(data, ' WIND_SPEED', ' ROTOR_SPEED')
box_plot(data, "Date", ' WIND_SPEED', ' V_OUT')
avg_plot(data, "Date", ' WIND_SPEED', ' V_OUT', )
wrose(data, ' WIND_SPEED', ' WIND_DIRECTION', 4)
frame_month = sep_df(data, "Date")
for x in frame_month:
    wrose(x, ' WIND_SPEED', ' WIND_DIRECTION', 4)