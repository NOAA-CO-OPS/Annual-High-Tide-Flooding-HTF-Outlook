import pandas as pd
import requests
import datetime as dt
import os
import  configparser
from calendar import isleap

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
run_year = data_params['run_year']

# Obtain historical High Tide Flooding by Met Year from NOAA COOPS API

met_yr_url = 'https://api.tidesandcurrents.noaa.gov/dpapi/prod/webapi/htf/htf_met_year_annual.xml'

met_yr_resp = requests.get(met_yr_url).content

met_yr_df = pd.read_xml(met_yr_resp)

met_yr_df = met_yr_df.drop(columns=['count'])

# Drops any rows with no station ID

met_yr_df = met_yr_df.dropna(subset=['stnId'])

met_yr_df = met_yr_df.reset_index(drop=True)


def calculate_percent_completeness(row):
    try:
        year = int(row['metYear'])
        nan_count = float(row['nanCount']) if pd.notna(row['nanCount']) else 0

        # Met year spans May YEAR to April (YEAR + 1)
        # So Feb of YEAR+1 determines leap day presence
        days_in_year = 366 if isleap(year + 1) else 365

        if round(nan_count) >= days_in_year:
            return None  # leave blank if fully missing

        completeness = round((days_in_year - nan_count) * 100 / days_in_year, 1)
        return completeness
    except:
        return None




met_yr_df['Percent Completeness'] = met_yr_df.apply(calculate_percent_completeness, axis=1)

cols_to_blank = ['majCount', 'modCount', 'minCount', 'Percent Completeness']

def blank_if_incomplete(row):
    try:
        if pd.notna(row['Percent Completeness']) and row['Percent Completeness'] < 85:
            min_count = float(row['minCount']) if pd.notna(row['minCount']) else 0
            if min_count == 0:
                for col in cols_to_blank:
                    row[col] = None
    except:
        pass
    return row

# Apply the blanking function to each row
#met_yr_df = met_yr_df.apply(blank_if_incomplete, axis=1)




# Saves HTF counts as a csv 

met_yr_df.to_csv(f'{data_dir}\\met_historic_flood_cts.csv',index=False)

print(f"Historic Flood Counts CSV created as {data_dir}\\met_historic_flood_cts.csv")
