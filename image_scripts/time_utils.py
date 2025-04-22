import datetime

def calc_date(ref_date, days_in_the_past):
    time_delay = datetime.timedelta(days=days_in_the_past)
    return ref_date - time_delay

# beginning_date = datetime.datetime(2014, 2, 16, 22, 30)
beginning_date = datetime.datetime(2015, 8, 19, 12, 0)
right_now = datetime.datetime.now()
one_day__in_past = calc_date(right_now, 1)
one_week_in_past = calc_date(right_now, 7)
one_mnth_in_past = calc_date(right_now, 30)
one_year_in_past = calc_date(right_now, 365)
time_wins_str = ['One_Day', 'One_Week', 'One_Month', 'One_Year']
time_wins = [one_day__in_past, one_week_in_past, one_mnth_in_past, one_year_in_past]
this_year = right_now.year

def get_time_bin(win_str):
    if time_wins_str[0] == win_str:
        time_bin_size = datetime.timedelta(hours=0.125)
    elif time_wins_str[1] == win_str:
        time_bin_size = datetime.timedelta(days=0.125)
    elif time_wins_str[2] == win_str:
        time_bin_size = datetime.timedelta(days=7./8.)
    elif time_wins_str[3] == win_str:
        time_bin_size = datetime.timedelta(days=15./4.)
    else:
        print('NO TIME BINS!')
    return time_bin_size