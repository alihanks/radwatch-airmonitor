import re
import pandas as pd
import datetime
import math
import os
import argparse
import time

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

# Delay between requests to avoid hammering WeatherUnderground
REQUEST_DELAY_SEC = 2


def get_csv_dates():
    """Read all unique dates present in weatherhawk.csv."""
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return set()
    try:
        df = pd.read_csv(CSV_PATH, usecols=[1], header=0)
        if len(df) == 0:
            return set()
        dates = pd.to_datetime(df.iloc[:, 0], errors='coerce').dropna()
        return set(dates.dt.date)
    except Exception as e:
        print(f"Could not read CSV dates: {e}")
        return set()


def get_last_csv_date():
    """Read the last timestamp in weatherhawk.csv and return its date."""
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return None
    try:
        df = pd.read_csv(CSV_PATH, usecols=[1], header=0)
        if len(df) == 0:
            return None
        last_val = df.iloc[-1, 0]
        last_dt = pd.to_datetime(last_val)
        return last_dt.date()
    except Exception as e:
        print(f"Could not determine last CSV date: {e}")
        return None


def get_first_csv_date():
    """Read the first timestamp in weatherhawk.csv and return its date."""
    if not os.path.exists(CSV_PATH) or os.path.getsize(CSV_PATH) == 0:
        return None
    try:
        df = pd.read_csv(CSV_PATH, usecols=[1], header=0, nrows=1)
        if len(df) == 0:
            return None
        first_val = df.iloc[0, 0]
        first_dt = pd.to_datetime(first_val)
        return first_dt.date()
    except Exception as e:
        print(f"Could not determine first CSV date: {e}")
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


def scrape_days(dates_to_fetch):
    """Scrape a list of dates and append to CSV. Returns total rows appended."""
    total_rows = 0
    for i, date in enumerate(sorted(dates_to_fetch)):
        if i > 0:
            time.sleep(REQUEST_DELAY_SEC)
        df = scrape_day(date)
        if df is not None and len(df) > 0:
            df.to_csv(CSV_PATH, mode='a', header=False, index=False)
            total_rows += len(df)
            print(f"  {date}: {len(df)} rows appended")
    return total_rows


def gather_data():
    """Scrape weather data for all missing days since last CSV entry through today."""
    today = datetime.date.today()
    last_date = get_last_csv_date()

    if last_date is None:
        print("No existing weather data found, fetching today only")
        start_date = today
    elif last_date >= today:
        print(f"Weather data already current (last entry: {last_date})")
        start_date = today  # Still re-fetch today to get latest readings
    else:
        start_date = last_date + datetime.timedelta(days=1)
        gap_days = (today - last_date).days
        print(f"Last weather data: {last_date}, backfilling {gap_days} day(s)")

    dates = []
    date = start_date
    while date <= today:
        dates.append(date)
        date += datetime.timedelta(days=1)

    total_rows = scrape_days(dates)
    print(f"Weather gathering complete: {total_rows} total rows appended")


def fill_gaps(since_date=None):
    """Find and fill internal gaps in weatherhawk.csv.

    Scans all dates from the first to the last entry (or from since_date
    to the last entry if provided) and scrapes any dates that are missing
    from the CSV. Appended data will be out of chronological order — run
    resort_weather_timestamps() afterward (raw_analysis.py does this
    automatically).
    """
    first_date = get_first_csv_date()
    last_date = get_last_csv_date()

    if first_date is None or last_date is None:
        print("Cannot fill gaps: CSV is empty or unreadable")
        return

    if since_date is not None and since_date > first_date:
        print(f"--since {since_date}: ignoring gaps before this date (CSV first entry is {first_date})")
        scan_start = since_date
    else:
        scan_start = first_date

    existing_dates = get_csv_dates()
    print(f"CSV date range: {first_date} to {last_date}")
    print(f"Scan range:     {scan_start} to {last_date}")
    print(f"Dates with data in scan range: {sum(1 for d in existing_dates if scan_start <= d <= last_date)}")

    # Find all missing dates in the scan range
    missing = []
    date = scan_start
    while date <= last_date:
        if date not in existing_dates:
            missing.append(date)
        date += datetime.timedelta(days=1)

    if not missing:
        print("No gaps found")
        return

    print(f"Found {len(missing)} missing date(s): {missing[0]} to {missing[-1]}")
    total_rows = scrape_days(missing)
    print(f"Gap filling complete: {total_rows} total rows appended across {len(missing)} day(s)")
    if total_rows > 0:
        print("NOTE: Data was appended out of order. The pipeline will re-sort")
        print("automatically (resort_weather_timestamps), or run manually:")
        print("  python3 -c \"from image_scripts.weather_utils import resort_weather_timestamps; "
              "resort_weather_timestamps('/home/dosenet/radwatch-airmonitor/weatherhawk.csv', "
              "'/home/dosenet/radwatch-airmonitor/data/weather_sorted.csv')\"")


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
    parser = argparse.ArgumentParser(description='Weather data gatherer for RadWatch')
    parser.add_argument('--fill-gaps', action='store_true',
                        help='Scan CSV for internal date gaps and backfill them from WeatherUnderground')
    parser.add_argument('--since', type=lambda s: datetime.date.fromisoformat(s), default=None,
                        metavar='YYYY-MM-DD',
                        help='With --fill-gaps, ignore gaps before this date (e.g. to skip stale pre-station-install entries)')
    args = parser.parse_args()

    if args.fill_gaps:
        fill_gaps(since_date=args.since)
    else:
        if args.since is not None:
            print("Note: --since is only used with --fill-gaps; ignoring")
        gather_data()
