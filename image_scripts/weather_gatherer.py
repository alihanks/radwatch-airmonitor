import re
import pandas as pd
import datetime
import math

CSV_PATH = '/home/dosenet/radwatch-airmonitor/Dropbox/UCB Air Monitor/Data/Weather/weatherhawk.csv'

NSEW_TO_DEG = {'N':180,'S':0,'E':270,'W':90,
               'North':180,'South':0,'East':270,'West':90,
               'NW':135,'NE':225,'SE':315,'SW':45,
               'NNW':157.5,'NNE':202.5,
               'ENE':247.5,'ESE':292.5,
               'SSE':337.5,'SSW':22.5,
               'WNW':112.5,'WSW':67.5}

def gather_data():
    #data grabbing
    date = datetime.date.today()
    web = "https://www.wunderground.com/dashboard/pws/{}/table/{}/{}/daily"
    station = "KCABERKE272"
    url = web.format(station,date,date)
    data_frame = pd.read_html(url,header=0)
    print(data_frame[1])
    data_frame = data_frame[3].iloc[1:]

    # wind direction conversion
    pd.set_option('future.no_silent_downcasting', True) # to suppress behavior warning
    data_frame.loc[:, 'Wind'] = data_frame.loc[:, 'Wind'].replace(NSEW_TO_DEG)

    # time conversion
    data_frame['Time'] = pd.to_datetime(str(date) + ' ' + data_frame['Time'], format='%Y-%m-%d %I:%M %p')

    # prevent existing data from being added again
    last_timestamp = get_last_timestamp()
    data_frame = data_frame[data_frame['Time'] > last_timestamp]

    # clean data of units
    measurement_cols = list(set(data_frame.columns) - {'Time'}) # excluding columns for measurement extraction
    data_frame[measurement_cols] = data_frame[measurement_cols].map(get_measurement)

    # assemble and append
    data_frame = assemble(data_frame)
    data_frame.to_csv(CSV_PATH, mode='a', header=False, index=False)

    # printout
    print(data_frame.tail())

def get_measurement(entry):
    match = re.search(r'[-+]?\d*\.?\d+', str(entry))        # using regex to extract the measurement
    return float(match.group()) if match else entry

def assemble(df):
    dict = {}
    dict['Record Id'] = -1                      # default val for unspecified entry num
    dict['Date Time'] = df['Time']
    dict['Air Temp Avg'] = df['Temperature']
    dict['Air Temp Min'] = df['Temperature']
    dict['Air Temp Min Time'] = df['Time']
    dict['Air Temp Max'] = df['Temperature']    # seems min, max, avg temp are shared (is it a single data point?)
    dict['Air Temp Max Time'] = df['Time']      # this info unavailable, using current time as proxy
    dict['Humidity'] = df['Humidity']
    dict['Barometer'] = df['Pressure']
    dict['Battery'] = math.nan
    dict['MinBattery'] = math.nan               # these labels don't seem to correspond with anything
    dict['ETo'] = math.nan
    dict['Rain Yearly'] = df['Precip. Accum.']              # a daily precip. accum. measurement exists
    dict['Solar Avg'] = df['Solar']             # this in w/m^2, does it require conversion?
    dict['Wind Speed Avg'] = df['Speed']
    dict['Wind Speed Max'] = df['Gust']         # assuming gust = max wind speed
    dict['Wind Speed Max Time'] = df['Time']    # this info unavailable, using current time as proxy
    dict['Wind Speed Avg 2'] = df['Speed']      # a duplicate column exists for avg wind speed
    dict['Wind Direction'] = df['Wind']

    df = pd.DataFrame(dict)
    return df

def get_last_timestamp():       # this determines the datetime of the last entry in the csv.
    for chunk in pd.read_csv(CSV_PATH, chunksize=1000, usecols=['Date Time']):
        val = chunk['Date Time'].iloc[-1]
        if val is not None:
            out = val
    return pd.Timestamp(out)

if(__name__=='__main__'):
    gather_data()