import pandas as pd
import datetime

NSEW_TO_DEG = {'N':180,'S':0,'E':270,'W':90,
               'North':180,'South':0,'East':270,'West':90,
               'NW':135,'NE':225,'SE':315,'SW':45,
               'NNW':157.5,'NNE':202.5,
               'ENE':247.5,'ESE':292.5,
               'SSE':337.5,'SSW':22.5,
               'WNW':112.5,'WSW':67.5}

def gather_data():
    date = datetime.date.today()
    web = "https://www.wunderground.com/dashboard/pws/{}/table/{}/{}/daily"
    station = "KCABERKE272"
    url = web.format(station,date,date)
    data_frame = pd.read_html(url,header=0)
    return reformat_cardinals(data_frame)

def reformat_cardinals(data_frame):
    data_frame[1].loc[2, :] = data_frame[1].loc[2, :].replace(NSEW_TO_DEG)
    data_frame[3].loc[:, 'Wind'] = data_frame[3].loc[:, 'Wind'].map(NSEW_TO_DEG)

    return data_frame

if(__name__=='__main__'):
    df = gather_data()
    print(df[1], df[3].head(5), sep='\n'*4)