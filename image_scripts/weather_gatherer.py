import re
import pandas as pd
import datetime
import math
import os

CSV_PATH = '/home/dosenet/radwatch-airmonitor/weatherhawk.csv'

NSEW_TO_DEG = {'N':180,'S':0,'E':270,'W':90,
               'North':180,'South':0,'East':270,'West':90,
               'NW':135,'NE':225,'SE':315,'SW':45,
               'NNW':157.5,'NNE':202.5,
               'ENE':247.5,'ESE':292.5,
               'SSE':337.5,'SSW':22.5,
               'WNW':112.5,'WSW':67.5}

WU_URL = "https://www.wunderground.com/dashboard/pws/{}/table/{}/{}/daily"
WU_STATION = "KCABERKE272"


def get_last_csv_date():
    """Read the last timestamp in weatherhawk.csv and return its date."""
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return None
    try:
        # Read just the second column (Date Time) to find the last entry
        df = pd.read_csv(CSV_PATH, usecols=[1], header=0)
        if len(df) == 0:
            return None
        last_val = df.iloc[-1, 0]
        last_dt = pd.to_datetime(last_val)
        return last_dt.date()
    except Exception as e:
        print(f"Could not determine last CSV date: {e}")
        return None


def scrape_day(date):
    """Scrape weather data for a single date from WeatherUnderground.

    Returns a DataFrame in the weatherhawk.csv format, or None on failure.
    """
    url = WU_URL.format(WU_STATION, date, date)
    print(f"Fetching weather for {date}: {url}")

    try:
        tables = pd.read_html(url, header=0)
    except Exception as e:
        print(f"Failed to fetch {date}: {e}")
        return None

    if len(tables) < 4:
        print(f"Unexpected table count for {date}: {len(tables)} (expected >= 4)")
        return None

    data_frame = tables[3].iloc[1:]

    if len(data_frame) == 0:
        print(f"No data rows for {date}")
        return None

    # wind direction conversion (NaN/missing values become NaN degrees)
    data_frame['Wind'] = data_frame['Wind'].map(
        lambda v: NSEW_TO_DEG.get(str(v), math.nan) if pd.notna(v) else math.nan
    )

    # time conversion
    data_frame['Time'] = pd.to_datetime(
        str(date) + ' ' + data_frame['Time'],
        format='%Y-%m-%d %I:%M %p'
    )

    # clean data of units
    measurement_cols = list(set(data_frame.columns) - {'Time'})
    data_frame[measurement_cols] = data_frame[measurement_cols].map(get_measurement)

    # assemble into weatherhawk.csv format
    return assemble(data_frame)


def gather_data():
    """Scrape weather data for all missing days since last CSV entry through today."""
    today = datetime.date.today()
    last_date = get_last_csv_date()

    if last_date is None:
        # No existing data — just get today
        print("No existing weather data found, fetching today only")
        start_date = today
    elif last_date >= today:
        # Already up to date
        print(f"Weather data already current (last entry: {last_date})")
        start_date = today  # Still re-fetch today to get latest readings
    else:
        # Backfill from the day after the last entry
        start_date = last_date + datetime.timedelta(days=1)
        gap_days = (today - last_date).days
        print(f"Last weather data: {last_date}, backfilling {gap_days} day(s)")

    # Scrape each missing day
    date = start_date
    total_rows = 0
    while date <= today:
        df = scrape_day(date)
        if df is not None and len(df) > 0:
            df.to_csv(CSV_PATH, mode='a', header=False, index=False)
            total_rows += len(df)
            print(f"  {date}: {len(df)} rows appended")
        date += datetime.timedelta(days=1)

    print(f"Weather gathering complete: {total_rows} total rows appended")


def get_measurement(entry):
    match = re.search(r'[-+]?\d*\.?\d+', str(entry))
    return float(match.group()) if match else entry


def assemble(df):
    result = {}
    result['Record Id'] = -1
    result['Date Time'] = df['Time']
    result['Air Temp Avg'] = df['Temperature']
    result['Air Temp Min'] = df['Temperature']
    result['Air Temp Min Time'] = df['Time']
    result['Air Temp Max'] = df['Temperature']
    result['Air Temp Max Time'] = df['Time']
    result['Humidity'] = df['Humidity']
    result['Barometer'] = df['Pressure']
    result['Battery'] = math.nan
    result['MinBattery'] = math.nan
    result['ETo'] = math.nan
    result['Rain Yearly'] = math.nan
    result['Solar Avg'] = df['Solar']
    result['Wind Speed Avg'] = df['Speed']
    result['Wind Speed Max'] = df['Gust']
    result['Wind Speed Max Time'] = df['Time']
    result['Wind Speed Avg 2'] = df['Speed']
    result['Wind Direction'] = df['Wind']

    return pd.DataFrame(result)


if __name__ == '__main__':
    gather_data()
