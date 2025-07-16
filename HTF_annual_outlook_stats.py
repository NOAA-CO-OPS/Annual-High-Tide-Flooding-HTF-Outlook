
import requests
import time as time
import pandas as pd
import numpy as np
import ast
import warnings
import configparser

#this is to suppress user warnings
warnings.filterwarnings("ignore")

# %% Configuration file
CONFIG_FILE = "config.cfg"

def read_config_section(config_file, section):
    params = {}
    try:
        config = configparser.ConfigParser()
        with open(config_file) as f:
            #config.readfp(f)
            config.read_file(f)
            options = config.options(section)
            for option in options:            
                try:
                    params[option] = config.get(section, option)
                    if params[option] == -1:
                        print("Could not read option: %s" % option)                    
                except:
                    print("Exception reading option %s!" % option)
                    params[option] = None
    except configparser.NoSectionError as nse:
        print("No section %s found reading %s: %s", section, config_file, nse)
    except IOError as ioe:
        print("Config file not found: %s: %s", config_file, ioe)

    return params

# Obtain directory info from config file
data_params = read_config_section(CONFIG_FILE, "dir")
work_dir = data_params['work_dir']
run_dir = data_params['run_dir']
data_dir = data_params['data_dir']
stn_list = data_params['stn_list_file']
analysis_year = int(data_params['run_year'])


#read in external data files
model_output = pd.read_csv(data_dir+'\\'+str(analysis_year)+'_HTF_Annual_Outlook_full_data.csv',dtype={'stnID': str})
model_output['Station_ID'] = model_output.stnID.str[:7]

lats_lons_df = pd.read_csv(stn_list)
lats_lons_df['NOAA_ID'] = lats_lons_df['NOAA_ID'].astype(str)

model_output = pd.merge(model_output,lats_lons_df,left_on='Station_ID',right_on='NOAA_ID')

station_list = model_output['Station_ID']

HTF_Annual_outlook_station_stats = model_output[['stnName','Station_ID','Region','projectMethod']]


#########################################################################################################
# functions
def quadratic_enso(x, z):
    return (a * x ** 2) + (b * x * z) + (c * x) + (d * z ** 2) + (e * z) + f
def quadratic(x):
    return (a * x ** 2) + (c * x) + f
def linear_enso(x,z):
    return (a * x) + (b*z) + c
def linear(x):
    return (a * x) + c
def no_t_lin_enso(z):
    return (a * z) + b
def no_t_quad_enso(z):
    return (a * z **2) + (b * z) + c
###########################################################################################################

#get previous year prediction
product = 'htf_met_year_annual_outlook'
met_year = str(int(analysis_year - 1))
server = 'https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/'

myurl = server+product+'.json?&met_year='+met_year

#https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/htf_met_year_annual_outlook.json?&units=english

    # Read JSON file
urlResponse = requests.get(myurl)
content=urlResponse.json()

    # Convert JSON encoded data into Python dictionary
mydata = content['MetYearAnnualOutlook']

past_pred = pd.DataFrame(mydata)

conf_cols = past_pred[['stnId', 'highConf', 'lowConf']].rename(columns={
    'highConf': 'prev_highConf',
    'lowConf': 'prev_lowConf'
})
HTF_Annual_outlook_station_stats['Station_ID'] = HTF_Annual_outlook_station_stats['Station_ID'].astype(str)
conf_cols['stnId'] = conf_cols['stnId'].astype(str)

HTF_Annual_outlook_station_stats = HTF_Annual_outlook_station_stats.merge(
    conf_cols,
    how='left',
    left_on='Station_ID',
    right_on='stnId'
).drop(columns='stnId')

#HTF_Annual_outlook_station_stats['prev_highConf'] = past_pred['highConf']
#HTF_Annual_outlook_station_stats['prev_lowConf'] = past_pred['lowConf']
HTF_Annual_outlook_station_stats['prev_mid'] = HTF_Annual_outlook_station_stats[['prev_highConf','prev_lowConf']].mean(axis=1)

#find the historical maximum number of annual HTF days
station_list = HTF_Annual_outlook_station_stats['Station_ID']
last_year_df = pd.DataFrame()
historic_df = pd.DataFrame()

for i in station_list:
  #get previous year prediction
  product = 'htf_met_year_annual'
  server = 'https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/'

  myurl = server+product+'.json?station='+str(i)

  urlResponse = requests.get(myurl)
  content=urlResponse.json()

  mydata = content['MetYearAnnualFloodCount']

  station_annual = pd.DataFrame(mydata)
  
  last_year_obs = station_annual[station_annual['metYear']==int(met_year)]
  last_year_df = pd.concat([last_year_df, last_year_obs], ignore_index=True)


  historic_max = station_annual['minCount'].idxmax()
  historic_max_year = station_annual.iloc[historic_max]

  historic_df = pd.concat([historic_df, historic_max_year.to_frame().T], ignore_index=True)

  last_year_df=last_year_df.reset_index(drop=True)

  historic_df=historic_df.reset_index(drop=True)

HTF_Annual_outlook_station_stats['prev_observed'] = last_year_df['minCount']

print(HTF_Annual_outlook_station_stats.dtypes)
HTF_Annual_outlook_station_stats['range'] = np.nan  # Start with all blank

for i in range(len(station_list)):
    obs = HTF_Annual_outlook_station_stats['prev_observed'].iloc[i]
    low = HTF_Annual_outlook_station_stats['prev_lowConf'].iloc[i]
    high = HTF_Annual_outlook_station_stats['prev_highConf'].iloc[i]

    # Skip row if any of the needed values are missing
    if pd.isna(obs) or pd.isna(low) or pd.isna(high):
        continue

    if obs < low:
        HTF_Annual_outlook_station_stats.at[i, 'range'] = 'BELOW'
    elif obs > high:
        HTF_Annual_outlook_station_stats.at[i, 'range'] = 'ABOVE'
    elif low <= obs <= high:
        HTF_Annual_outlook_station_stats.at[i, 'range'] = 'WITHIN'

model_output['mid_pred'] = (model_output['highConf']+model_output['lowConf'])/2
HTF_new_predictions = model_output[['lowConf','highConf','mid_pred']]

HTF_Annual_outlook_station_stats['current_highConf'] = HTF_new_predictions['highConf']
HTF_Annual_outlook_station_stats['current_lowConf'] = HTF_new_predictions['lowConf']
HTF_Annual_outlook_station_stats['current_mid'] = HTF_new_predictions['mid_pred']
#find historic max HTF count per station
HTF_Annual_outlook_station_stats['historic_max_HTF_days'] = historic_df['minCount']
HTF_Annual_outlook_station_stats['historic_met_year'] = historic_df['metYear']
#check if last observed broke station record
HTF_Annual_outlook_station_stats['record_break'] = np.nan

for i in range(len(station_list)):
    obs = HTF_Annual_outlook_station_stats['prev_observed'].iloc[i]
    hist_max = HTF_Annual_outlook_station_stats['historic_max_HTF_days'].iloc[i]

    if pd.isna(obs):
        HTF_Annual_outlook_station_stats.at[i, 'record_break'] = None  # leave blank
    elif obs < hist_max:
        HTF_Annual_outlook_station_stats.at[i, 'record_break'] = 'NO'
    else:
        HTF_Annual_outlook_station_stats.at[i, 'record_break'] = 'YES'

HTF_Annual_outlook_station_stats['2000_trend_only_days'] = model_output['pred_2k']

trend_only = pd.DataFrame(columns=['trend_only_highConf','trend_only_lowConf','trend_only_mid'])
#find trend only pred for enso stations
for i in range(len(model_output)):
    station = model_output.loc[i,'Station_ID']
    station_df = model_output.loc[model_output['Station_ID']==station]
    
    if station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
        x=station_df['lin_pred'].iloc[0]
        x_high=x+station_df['lin_rmse'].iloc[0]
        x_low=x-station_df['lin_rmse'].iloc[0]
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
        x=station_df['quad_pred'].iloc[0]
        x_high=x+station_df['quad_rmse'].iloc[0]
        x_low=x-station_df['quad_rmse'].iloc[0]
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Linear':
        x = np.nan
        x_high = np.nan
        x_low=np.nan
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'Quadratic':
        x = np.nan
        x_high = np.nan
        x_low=np.nan
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)

    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Linear ENSO Sensitivity':
        x=station_df['avg19'].iloc[0]
        x_high=x+station_df['stdev19'].iloc[0]
        x_low=x-station_df['stdev19'].iloc[0]
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Quadratic ENSO Sensitivity':
        x=station_df['avg19'].iloc[0]
        x_high=x+station_df['stdev19'].iloc[0]
        x_low=x-station_df['stdev19'].iloc[0]
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)
        
    else:
        x = np.nan
        x_high = np.nan
        x_low=np.nan
        trend_only = pd.concat([trend_only, pd.DataFrame({'trend_only_highConf': [x_high],'trend_only_lowConf': [x_low],'trend_only_mid': [x]})], ignore_index=True)

HTF_Annual_outlook_station_stats['trend_only_highConf'] = trend_only['trend_only_highConf']
HTF_Annual_outlook_station_stats['trend_only_lowConf'] = trend_only['trend_only_lowConf']
HTF_Annual_outlook_station_stats['trend_only_mid'] = trend_only['trend_only_mid']

HTF_Annual_outlook_station_stats['pred_difference'] = HTF_Annual_outlook_station_stats['current_mid'] - HTF_Annual_outlook_station_stats['trend_only_mid']

HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'] = HTF_Annual_outlook_station_stats['2000_trend_only_days']
HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'] = HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'].where(HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'] > 0, 0.5)

HTF_Annual_outlook_station_stats['percent_increase_trend_only_since_2000'] = ((HTF_Annual_outlook_station_stats['trend_only_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only_nozeros']) / HTF_Annual_outlook_station_stats['2000_trend_only_nozeros']*100)

HTF_Annual_outlook_station_stats['percent_increase_chosen_methods_since_2000'] = ((HTF_Annual_outlook_station_stats['current_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only_nozeros']) / HTF_Annual_outlook_station_stats['2000_trend_only_nozeros']*100)

HTF_Annual_outlook_station_stats['days_increase_since_2000'] = ((HTF_Annual_outlook_station_stats['current_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only_nozeros'])).round()

HTF_Annual_outlook_station_stats=HTF_Annual_outlook_station_stats.drop(columns=['2000_trend_only_nozeros'])

#4/25/2025 adding previous year statistics
#read in enso info
enso_oni = pd.read_csv(data_dir+'\\'+'annual_means.csv')
avg_file = pd.read_csv(data_dir+'\\'+'average_all_models.txt',header=None)
#pull historic counts from data file
all_obs_df = pd.read_csv(data_dir+'\\'+'met_historic_flood_cts.csv')
all_obs_df['station_ID'] = all_obs_df['stnId'].astype(str).str[:7]
forecast_avg = avg_file.iloc[0][0]
enso_oni.loc[len(enso_oni)] = [int(met_year), np.nan, forecast_avg, np.nan]
enso_oni = enso_oni['ONI Met Year'].groupby(enso_oni['Year']).mean()

station_past = pd.DataFrame()

for i in range(len(model_output)):
  station_id = model_output.loc[i,'Station_ID']

  obs_df=all_obs_df[all_obs_df['station_ID']==station_id]

  obs_df=obs_df[obs_df['metYear']>1949]

  obs = obs_df.dropna(subset=['minCount'])

  obs['HTF']='days per year'

  data_start_year = obs['metYear'].iloc[0]

################################################################################
  station_df = model_output.loc[model_output['Station_ID']==station_id]

  station_method = station_df['projectMethod']

#for statiosn with a project method with ENSO sensitivity, we also want to grab the time-based method as well
  if station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
          trend_only = station_df[['stnID','stnName','lin_pred','lin_rmse']]
          trend_only['Low_Pred'] = trend_only['lin_pred'] - trend_only['lin_rmse']
          trend_only['High_Pred'] = trend_only['lin_pred'] + trend_only['lin_rmse']

  elif station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
          trend_only = station_df[['stnID','stnName','quad_pred','quad_rmse']]
          trend_only['Low_Pred'] = trend_only['quad_pred'] - trend_only['quad_rmse']
          trend_only['High_Pred'] = trend_only['quad_pred'] + trend_only['quad_rmse']

  elif station_df['projectMethod'].iloc[0] in ['No Temporal Trend with Linear ENSO Sensitivity', 'No Temporal Trend with Quadratic ENSO Sensitivity']:
          trend_only = station_df[['stnID','stnName','avg19','stdev19']]
          trend_only['Low_Pred'] = trend_only['avg19'] - trend_only['stdev19']
          if trend_only['Low_Pred'].iloc[0] < 0:
              trend_only['Low_Pred'] = 0
          trend_only['High_Pred'] = trend_only['avg19'] + trend_only['stdev19']

  predict_df = station_df[['stnID','stnName','lowConf','highConf']]
  predict_df.rename(columns={'lowConf':'Low_Pred','highConf':'High_Pred'},inplace=True)

      #######################################################################################

      #getting trend formulas
  if station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
          station_formula = station_df['lin_enso_fit'].iloc[0]
          station_trend = station_df['lin_fit'].iloc[0]
          coefficients = ast.literal_eval(station_formula)
          coefficients_trend = ast.literal_eval(station_trend)
  elif station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
          station_formula = station_df['quad_enso_fit'].iloc[0]
          station_trend = station_df['quad_fit'].iloc[0]
          coefficients = ast.literal_eval(station_formula)
          coefficients_trend = ast.literal_eval(station_trend)
  elif station_df['projectMethod'].iloc[0] == 'Linear':
          station_formula = station_df['lin_fit'].iloc[0]
          coefficients = ast.literal_eval(station_formula)
  elif station_df['projectMethod'].iloc[0] == 'Quadratic':
          station_formula = station_df['quad_fit'].iloc[0]
          coefficients = ast.literal_eval(station_formula)
  elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Linear ENSO Sensitivity':
          station_formula = station_df['no_t_lin_fit'].iloc[0]
          station_trend = station_df['avg19'].iloc[0]
          coefficients = ast.literal_eval(station_formula)
          coefficients_avg = station_trend
  elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Quadratic ENSO Sensitivity':
          station_formula = station_df['no_t_quad_fit'].iloc[0]
          station_trend = station_df['avg19'].iloc[0]
          coefficients = ast.literal_eval(station_formula)
          coefficients_avg = station_trend
  else:
          station_formula = station_df['avg19']
          coefficients_avg = station_formula

      #######################################################################################

      #calculating trend lines
  data_length = int(met_year)-1949
  x_vals = np.linspace(1950, int(met_year),data_length)
  z = 0

          # Extract coefficients
  if station_df['projectMethod'].iloc[0] == 'Quadratic With ENSO Sensitivity':
          a = coefficients['np.power(x, 2)']
          b = coefficients['x:z']
          c = coefficients['x']
          d = coefficients['np.power(z, 2)']
          e = coefficients['z']
          f = coefficients['Intercept']
          y_vals = quadratic_enso(x_vals, z)
          y_vals_trend = quadratic_enso(x_vals,enso_oni)
          other_method = 'Quadratic'
  elif station_df['projectMethod'].iloc[0] =='Quadratic':
          a = coefficients['np.power(x, 2)']
          c = coefficients['x']
          f = coefficients['Intercept']
          y_vals_trend = quadratic(x_vals)
  elif station_df['projectMethod'].iloc[0] == 'Linear With ENSO Sensitivity':
          a = coefficients['x']
          b = coefficients['z']
          c = coefficients['Intercept']
          y_vals = linear_enso(x_vals, z)
          y_vals_trend = linear_enso(x_vals,enso_oni)
          other_method = 'Linear'
  elif station_df['projectMethod'].iloc[0] == 'Linear':
          a = coefficients['x']
          c = coefficients['Intercept']
          y_vals_trend = linear(x_vals)
  elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Linear ENSO Sensitivity':
          a = coefficients['z']
          b = coefficients['Intercept']
          y_vals_trend = no_t_lin_enso(enso_oni)
          y_vals = [coefficients_avg]*data_length
          other_method = '19-yr Average'
  elif station_df['projectMethod'].iloc[0] == 'No Temporal Trend with Quadratic ENSO Sensitivity':
          a = coefficients['np.power(z, 2)']
          b = coefficients['z']
          c = coefficients['Intercept']
          y_vals_trend = no_t_quad_enso(enso_oni)
          y_vals = [coefficients_avg]*data_length
          other_method = '19-yr Average'
  else:
          y_vals_trend = [coefficients_avg.iloc[0]]*data_length

  station_past_info = pd.DataFrame(columns=['stnID','2023_trend_only_value','2023_chosen_method_value','2023_obs'])
  station_past_info['stnID'] = [station_id]
  station_past_info['2023_obs'] = obs['minCount'].iloc[-1]
  if station_df['projectMethod'].iloc[0] in ['Linear With ENSO Sensitivity', 'Quadratic With ENSO Sensitivity',
                                            'No Temporal Trend with Linear ENSO Sensitivity',
                                            'No Temporal Trend with Quadratic ENSO Sensitivity']:
    station_past_info['2023_trend_only_value'] = y_vals[-1]
    station_past_info['2023_chosen_method_value'] = y_vals_trend.iloc[-1]
  else:
    station_past_info['2023_trend_only_value'] = np.nan
    station_past_info['2023_chosen_method_value'] = y_vals_trend[-1]

  station_past = pd.concat([station_past,station_past_info])

station_past = station_past.reset_index(drop=True)
station_past['2023_trend_only_value'] = station_past['2023_trend_only_value'].replace(0,0.5)
station_past['2023_chosen_method_value'] = station_past['2023_chosen_method_value'].replace(0,0.5)

station_past['enso_difference'] = station_past['2023_chosen_method_value'] - station_past['2023_trend_only_value']
station_past['enso_percent_increase'] = (station_past['enso_difference'] / station_past['2023_trend_only_value'])*100
station_past['2023_trend_only_value'].fillna(station_past['2023_chosen_method_value'], inplace=True)
station_past['obs_difference'] = station_past['2023_obs'] - station_past['2023_trend_only_value']
station_past['obs_percent_increase'] = (station_past['obs_difference'] / station_past['2023_trend_only_value'])*100

HTF_Annual_outlook_station_stats['2023_trend_only_value'] = station_past['2023_trend_only_value']
HTF_Annual_outlook_station_stats['2023_chosen_method_value'] = station_past['2023_chosen_method_value']
HTF_Annual_outlook_station_stats['2023_enso_difference'] = station_past['enso_difference']
HTF_Annual_outlook_station_stats['2023_enso_percent_increase'] = station_past['enso_percent_increase']
HTF_Annual_outlook_station_stats['2023_obs_trend_only_difference'] = station_past['obs_difference']
HTF_Annual_outlook_station_stats['2023_obs_trend_only_percent_increase'] = station_past['obs_percent_increase']


HTF_Annual_outlook_station_stats.to_csv(f'{data_dir}\\{analysis_year}_HTF_Annual_Outlook_station_stats.csv',index=False)

print('Station stats exported')
###################################################################################################################################################################
###################################################################################################################################################################

#region stats
regions = HTF_Annual_outlook_station_stats['Region'].unique()

HTF_Annual_outlook_region_stats = pd.DataFrame()

HTF_Annual_outlook_region_stats['Region'] = regions

additional_regions = ['US','CONUS']

additional_df = pd.DataFrame({'Region': additional_regions})

HTF_Annual_outlook_region_stats = pd.concat([HTF_Annual_outlook_region_stats, additional_df], ignore_index=True)

HTF_Annual_outlook_region_stats.index=HTF_Annual_outlook_region_stats['Region']

#Median value (over all stations within region) of last year’s observed HTF days
median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['prev_observed'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

conus_df = HTF_Annual_outlook_station_stats[~HTF_Annual_outlook_station_stats['Region'].isin(['PAC', 'CAR', 'AK'])]

median_conus = conus_df['prev_observed'].median()

median_us = HTF_Annual_outlook_station_stats['prev_observed'].median()

additional_regions = [median_us,median_conus]

additional_df = pd.DataFrame({'prev_observed': additional_regions},index=['US','CONUS'])

df = pd.concat([median_regions, additional_df])

HTF_Annual_outlook_region_stats['median_prev_observed'] = df

# Filter to only include stations with both prev_high and prev_low
valid_stations = HTF_Annual_outlook_station_stats.dropna(subset=['prev_highConf', 'prev_lowConf','prev_observed'])

# Total number of valid stations per region
num_station_region = pd.DataFrame(valid_stations['Station_ID'].groupby(valid_stations['Region']).count())

# Count of stations by range category per region
above = valid_stations[valid_stations['range'] == 'ABOVE']
num_above = pd.DataFrame(above['range'].groupby(above['Region']).count())

within = valid_stations[valid_stations['range'] == 'WITHIN']
num_within = pd.DataFrame(within['range'].groupby(within['Region']).count())

below = valid_stations[valid_stations['range'] == 'BELOW']
num_below = pd.DataFrame(below['range'].groupby(below['Region']).count())

# Add percentage columns
num_station_region['percent_above'] = round((num_above['range'] / num_station_region['Station_ID']) * 100, 2)
num_station_region['percent_within'] = round((num_within['range'] / num_station_region['Station_ID']) * 100, 2)
num_station_region['percent_below'] = round((num_below['range'] / num_station_region['Station_ID']) * 100, 2)


# === Filter CONUS for valid stations ===
valid_conus = conus_df.dropna(subset=['prev_highConf', 'prev_lowConf', 'prev_observed'])
conus_num_stations = len(valid_conus['Station_ID'].unique())

above = valid_conus[valid_conus['range'] == 'ABOVE']
num_above = above['range'].count()

within = valid_conus[valid_conus['range'] == 'WITHIN']
num_within = within['range'].count()

below = valid_conus[valid_conus['range'] == 'BELOW']
num_below = below['range'].count()

conus_percent_above = round((num_above / conus_num_stations) * 100, 2)
conus_percent_within = round((num_within / conus_num_stations) * 100, 2)
conus_percent_below = round((num_below / conus_num_stations) * 100, 2)

df_con = pd.DataFrame(index=['CONUS'])
df_con['Station_ID'] = conus_num_stations
df_con['percent_above'] = conus_percent_above
df_con['percent_within'] = conus_percent_within
df_con['percent_below'] = conus_percent_below

# Append CONUS to region summary
num_station_region = pd.concat([num_station_region, df_con])


# === US Total using only valid stations ===
valid_us = HTF_Annual_outlook_station_stats.dropna(subset=['prev_highConf', 'prev_lowConf', 'prev_observed'])
total_num_stations = len(valid_us['Station_ID'].unique())

above = valid_us[valid_us['range'] == 'ABOVE']
num_above = above['range'].count()

within = valid_us[valid_us['range'] == 'WITHIN']
num_within = within['range'].count()

below = valid_us[valid_us['range'] == 'BELOW']
num_below = below['range'].count()

total_percent_above = round((num_above / total_num_stations) * 100, 2)
total_percent_within = round((num_within / total_num_stations) * 100, 2)
total_percent_below = round((num_below / total_num_stations) * 100, 2)

df_tot = pd.DataFrame(index=['US'])
df_tot['Station_ID'] = total_num_stations
df_tot['percent_above'] = total_percent_above
df_tot['percent_within'] = total_percent_within
df_tot['percent_below'] = total_percent_below

# Append US total to region summary
num_station_region = pd.concat([num_station_region, df_tot])

# Fill any missing values with 0.0
num_station_region = num_station_region.fillna(0.0)

# Correct total number of stations per region (no filtering)
total_station_region = HTF_Annual_outlook_station_stats.groupby('Region')['Station_ID'].nunique().to_frame(name='Station_ID')

# Total US station count (ALL stations, no filtering)
total_us_stations = HTF_Annual_outlook_station_stats['Station_ID'].nunique()

# Total CONUS station count (ALL stations in CONUS, no filtering)
total_conus_stations = conus_df['Station_ID'].nunique()

total_station_region.loc['CONUS'] = total_conus_stations
total_station_region.loc['US'] = total_us_stations


# Update final region stats
HTF_Annual_outlook_region_stats.index = HTF_Annual_outlook_region_stats['Region']
HTF_Annual_outlook_region_stats['num_station_per_region'] = total_station_region['Station_ID']
HTF_Annual_outlook_region_stats['percent_above'] = num_station_region['percent_above']
HTF_Annual_outlook_region_stats['percent_within'] = num_station_region['percent_within']
HTF_Annual_outlook_region_stats['percent_below'] = num_station_region['percent_below']

#Median values of all station’s next year’s upper and lower model (trend+ENSO) predictions
median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['current_highConf'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

median_conus = conus_df['current_highConf'].median()

median_us = HTF_Annual_outlook_station_stats['current_highConf'].median()

additional_regions = [median_us, median_conus]

additional_df = pd.DataFrame({'current_highConf': additional_regions}, index=['US', 'CONUS'])

median_regions = pd.concat([median_regions, additional_df])


HTF_Annual_outlook_region_stats['median_current_highConf'] = median_regions

median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['current_lowConf'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

median_conus = conus_df['current_lowConf'].median()

median_us = HTF_Annual_outlook_station_stats['current_lowConf'].median()

additional_regions = [median_us,median_conus]

additional_df = pd.DataFrame({'current_lowConf': additional_regions}, index=['US', 'CONUS'])

median_regions = pd.concat([median_regions, additional_df])


HTF_Annual_outlook_region_stats['median_current_lowConf'] = median_regions

#Median value of number of days increase since year 2000 trend only to next year’s trend only prediction
HTF_Annual_outlook_station_stats['days_increase_trend_only'] = HTF_Annual_outlook_station_stats['trend_only_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only_days']

HTF_Annual_outlook_station_stats['days_increase_chosen_method'] = HTF_Annual_outlook_station_stats['current_mid'] - HTF_Annual_outlook_station_stats['2000_trend_only_days']

conus_df = HTF_Annual_outlook_station_stats[~HTF_Annual_outlook_station_stats['Region'].isin(['PAC', 'CAR', 'AK'])]

median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['days_increase_trend_only'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

median_conus = conus_df['days_increase_trend_only'].median()

median_us = HTF_Annual_outlook_station_stats['days_increase_trend_only'].median()

additional_regions = [median_us,median_conus]

additional_df = pd.DataFrame({'days_increase_trend_only': additional_regions}, index=['US', 'CONUS'])

median_regions = pd.concat([median_regions, additional_df])

HTF_Annual_outlook_region_stats['median_days_increase_trend_only'] = median_regions

median_regions = pd.DataFrame(HTF_Annual_outlook_station_stats['days_increase_chosen_method'].groupby(HTF_Annual_outlook_station_stats['Region']).median())

median_conus = conus_df['days_increase_chosen_method'].median()

median_us = HTF_Annual_outlook_station_stats['days_increase_chosen_method'].median()

additional_regions = [median_us, median_conus]

additional_df = pd.DataFrame({'days_increase_chosen_method': additional_regions}, index=['US', 'CONUS'])

median_regions = pd.concat([median_regions, additional_df])

HTF_Annual_outlook_region_stats['median_days_increase_chosen_method'] = median_regions

#Median of the station difference between previous two columns (to see regional influence of ENSO)
HTF_Annual_outlook_region_stats['median_days_increase_diff'] = HTF_Annual_outlook_region_stats['median_days_increase_trend_only']-HTF_Annual_outlook_region_stats['median_days_increase_chosen_method']

#Which station within region had the largest number of observed HTF days last year? What is that value?
max_regions = HTF_Annual_outlook_station_stats.loc[HTF_Annual_outlook_station_stats.groupby('Region')['prev_observed'].idxmax()]
max_regions = max_regions[['Station_ID','Region','prev_observed']]
max_regions.index=max_regions['Region']

max_conus = conus_df.loc[conus_df['prev_observed'].idxmax()]

max_conus_df = pd.DataFrame({'Station_ID': max_conus[1]}, index=['CONUS'])
max_conus_df['Region'] = max_conus[2]
max_conus_df['prev_observed'] = max_conus[7]

max_regions = pd.concat([max_regions, max_conus_df])


max_us = HTF_Annual_outlook_station_stats.loc[HTF_Annual_outlook_station_stats['prev_observed'].idxmax()]
max_us_df = pd.DataFrame({'Station_ID':max_us[1]},index=['US'])
max_us_df['Region']=max_us[2]
max_us_df['prev_observed']=max_us[7]

max_regions = pd.concat([max_regions, max_us_df])

HTF_Annual_outlook_region_stats['ID_for_prev_highest'] = max_regions['Station_ID']
HTF_Annual_outlook_region_stats['Highest_prev_observed'] = max_regions['prev_observed']

#Which station within region is predicted to have the largest number of observed HTF days next year?  What is that value?
max_regions = HTF_Annual_outlook_station_stats.loc[HTF_Annual_outlook_station_stats.groupby('Region')['current_mid'].idxmax()]
max_regions = max_regions[['Station_ID','Region','current_mid']]
max_regions.index=max_regions['Region']

max_conus = conus_df.loc[conus_df['current_mid'].idxmax()]
max_conus_df = pd.DataFrame({'Station_ID':max_conus[1]},index=['CONUS'])
max_conus_df['Region']=max_conus[2]
max_conus_df['current_mid']=max_conus[11]

max_regions = pd.concat([max_regions, max_conus_df])

max_us = HTF_Annual_outlook_station_stats.loc[HTF_Annual_outlook_station_stats['current_mid'].idxmax()]
max_us_df = pd.DataFrame({'Station_ID':max_us[1]},index=['US'])
max_us_df['Region']=max_us[2]
max_us_df['current_mid']=max_us[11]

max_regions = pd.concat([max_regions, max_us_df])

HTF_Annual_outlook_region_stats['ID_for_pred_highest'] = max_regions['Station_ID']
HTF_Annual_outlook_region_stats['Highest_predicted'] = max_regions['current_mid']

median_percent_increase = HTF_Annual_outlook_station_stats.groupby('Region')['percent_increase_chosen_methods_since_2000'].median()
median_conus = conus_df['percent_increase_chosen_methods_since_2000'].median()
median_us = HTF_Annual_outlook_station_stats['percent_increase_chosen_methods_since_2000'].median()
additional_regions = [median_us, median_conus]
additional_df = pd.DataFrame({'percent_increase_chosen_methods_since_2000': additional_regions}, index=['US', 'CONUS'])
median_percent_increase = pd.concat([median_percent_increase, additional_df])
HTF_Annual_outlook_region_stats['Percent_increase_chosen_methods_since_2000'] = median_percent_increase

median_days_increase_since_2000 = HTF_Annual_outlook_station_stats.groupby('Region')['days_increase_since_2000'].median()
median_conus = conus_df['days_increase_since_2000'].median()
median_us = HTF_Annual_outlook_station_stats['days_increase_since_2000'].median()
additional_regions = [median_us, median_conus]
additional_df = pd.DataFrame({'days_increase_since_2000': additional_regions}, index=['US', 'CONUS'])
median_days_increase_since_2000 = pd.concat([median_days_increase_since_2000, additional_df])
HTF_Annual_outlook_region_stats['Median_days_increase_since_2000'] = median_days_increase_since_2000

HTF_Annual_outlook_region_stats.to_csv(f'{data_dir}\\{analysis_year}_HTF_Annual_Outlook_region_stats.csv',index=False)
print('Regional stats exported')
print('End of script')
# %%
