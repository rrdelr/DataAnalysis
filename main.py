from funcs import to_df, info, gen_scat, wrose, sep_df, box_plot, avg_plot, gen_vs, heat, compare

# Main sequence
sheet_t1 = 'TURBINA-1'
df_t1 = to_df("PARAMETROS_PROTOTIPO(WIP10+ 2019).xlsx", sheet_t1)
# info(df_t1, ' WIND_SPEED', ' ROTOR_SPEED', ' V_OUT')
# gen_scat(df_t1, sheet_t1, '06', '11', 'Date', ' V_OUT')
# gen_scat(df_t1, sheet_t1, '06', '11', 'Date', ' WIND_SPEED')
# gen_scat(df_t1, sheet_t1, '06', '11', 'Date', ' ROTOR_SPEED')
# heat(df_t1[[' WIND_SPEED', ' ROTOR_SPEED', ' V_OUT', ' GRID_POWER', ' POWER_HZ']], sheet_t1)
# gen_vs(df_t1, sheet_t1,' ROTOR_SPEED', ' V_OUT')
# # #
# gen_vs(df_t1, sheet_t1,  ' ROTOR_SPEED', ' V_OUT')
# box_plot(df_t1, sheet_t1, "Date", ' WIND_SPEED')
# box_plot(df_t1, sheet_t1, "Date",' V_OUT')
# avg_plot(df_t1, sheet_t1, "Date", ' WIND_SPEED', ' ROTOR_SPEED')
# wrose(df_t1, sheet_t1, ' WIND_SPEED', ' WIND_DIRECTION', 6)
# frame_month = sep_df(df_t1, "Date")
# for x in frame_month:
#     wrose(x, sheet_t1, ' WIND_SPEED', ' WIND_DIRECTION', 6)

# Second sheet
sheet_t2 = 'TURBINA-2'
df_t2 = to_df("PARAMETROS_PROTOTIPO(WIP10+ 2019).xlsx", sheet_t2)
# info(df_t2, ' WIND_SPEED', ' ROTOR_SPEED', ' V_OUT')
# gen_scat(df_t2, sheet_t2, '06', '11', 'Date', ' V_OUT')
# gen_scat(df_t2, sheet_t2, '06', '11', 'Date', ' WIND_SPEED')
# gen_scat(df_t2, sheet_t2, '06', '11', 'Date', ' ROTOR_SPEED')
# heat(df_t2[[' WIND_SPEED', ' ROTOR_SPEED', ' V_OUT', ' GRID_POWER', ' POWER_HZ']], sheet_t2)
# gen_vs(df_t2, sheet_t2,' ROTOR_SPEED', ' V_OUT')
# # #
# gen_vs(df_t2, sheet_t2,  ' ROTOR_SPEED', ' V_OUT')
# box_plot(df_t2, sheet_t2, "Date", ' WIND_SPEED')
# box_plot(df_t2, sheet_t2, "Date",' V_OUT')
# avg_plot(df_t2, sheet_t2, "Date", ' WIND_SPEED', ' ROTOR_SPEED')
# wrose(df_t2, sheet_t2, ' WIND_SPEED', ' WIND_DIRECTION', 6)
# frame_month = sep_df(df_t2, "Date")
# for x in frame_month:
#     wrose(x, sheet_t2, ' WIND_SPEED', ' WIND_DIRECTION', 6)

# Comparison of two sheets
compare((df_t1, df_t2), '06', '11', ' WIND_SPEED')
compare((df_t1, df_t2), '06', '11', ' ROTOR_SPEED')
compare((df_t1, df_t2), '06', '11', ' V_OUT')