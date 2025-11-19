from itertools import chain
import calendar
import weather_utils
from time_utils import *
import numpy as np
import h5py
import datetime

import glob
import os
import sys
sys.path.insert(0, "..")
#import spectra_utils


class ROI:
    def __init__(self, roi_peak, roi_bkg):
        self.peak = roi_peak
        self.bkg = roi_bkg
        self.isotope = ''
        self.origin = ''
        self.counts = []
        self.energy = []
        self.errors = []

    def get_counts(self, spec):
        sum_ = 0
        k = 0
        for el in spec[self.peak[0]:self.peak[1]]:
            sum_ += el
            k += 1
        bkg_sum_ = 0
        l = 0
        for bkg_el in self.bkg:
            for p in spec[bkg_el[0]:bkg_el[1]]:
                l += 1
                bkg_sum_ += p
        self.counts = sum_ - float(bkg_sum_) / float(l) * float(k)
        self.error = np.sqrt(sum_ + float(bkg_sum_) / float(l) * float(k))
        return self.counts, self.error


class Sample:
    def __init__(self):
        self.live_time = datetime.timedelta(seconds=0)  # time of sampling
        self.real_time = datetime.timedelta(seconds=0)
        self.timestamp = datetime.datetime.now()  # beginning of acquisition
        # weather vals
        self.weather_timestamps = np.zeros(0)
        self.temp = np.zeros(0)  # temperature array over time stamp
        self.pressures = []
        self.solar = np.zeros(0)
        self.relh = np.zeros(0)
        self.wind_speed = np.zeros(0)
        self.wind_dir = np.zeros(0)
        self.rain = []
        # spectra vals
        self.bin_lim = np.zeros(2)
        self.bin_cal = np.zeros(2)
        self.energy = []
        self.counts = []
        self.eff_curve = []

    def __add__(self, other):
        self.live_time = self.live_time + other.live_time
        self.real_time = self.real_time + other.real_time
        # self.timestamp
        #
        self.weather_timestamps = np.hstack([self.weather_timestamps, other.weather_timestamps])
        self.temp = np.hstack([self.temp, other.temp])
        self.solar = np.hstack([self.solar, other.solar])
        self.relh = np.hstack([self.relh, other.relh])
        self.wind_speed = np.hstack([self.wind_speed, other.wind_speed])
        self.wind_dir = np.hstack([self.wind_dir, other.wind_dir])
        #
        self.counts = self.counts + other.counts
        return self

    def __str__(self):
        print_str = str(self.timestamp)
        print_str = print_str + '\n\tWeather Timestamps: ' + str(len(self.weather_timestamps))
        print_str = print_str + '\n\tTemperatures: ' + str(self.temp.shape)
        print_str = print_str + '\n\tPressures: ' + str(self.pressures.shape)
        print_str = print_str + '\n\tSolar: ' + str(self.solar.shape)
        print_str = print_str + '\n\tRel. Hum.: ' + str(self.relh.shape)
        print_str = print_str + '\n\tWind Speed: ' + str(self.wind_speed.shape)
        print_str = print_str + '\n\tWind Dir.: ' + str(self.wind_dir.shape)
        print_str = print_str + '\n\tLive Time: ' + str(self.live_time)
        print_str = print_str + '\n\tSpectra Channels: ' + str(self.bin_lim[1] - self.bin_lim[0] + 1)
        print_str = print_str + '\n\tEnergy Cal: ' + str(self.bin_cal[0]) + '+' + str(self.bin_cal[1]) + 'x'
        return print_str

    def set_weather_timestamps(self, weather_time):
        self.weather_timestamps = weather_time

    def set_temps(self, temperatures):
        self.temp = temperatures

    def set_pressures(self, pres):
        self.pressures = pres

    def set_solars(self, solars):
        self.solar = solars

    def set_relativehs(self, rhs):
        self.relh = rhs

    def set_wind_speeds(self, wind_s):
        self.wind_speed = wind_s

    def set_wind_dirs(self, dirs):
        self.wind_dir = dirs

    def set_rain(self, rains):
        self.rain = rains

    def set_weather_params(self, weather_timestamps, temperatures, pressures, solars, rhs, wind_s, wind_d, rain):
        if len(weather_timestamps) == 0:
            weather_timestamps = np.zeros(1)
            weather_timestamps[0] = np.nan
            temperatures = np.zeros(1)
            temperatures[0] = np.nan
            pressures = np.zeros(1)
            pressures[0] = np.nan
            solars = np.zeros(1)
            solars[0] = np.nan
            rhs = np.zeros(1)
            rhs[0] = np.nan
            wind_s = np.zeros(1)
            wind_s[0] = 0
            wind_d = np.zeros(1)
            wind_d[0] = 0

        self.set_weather_timestamps(weather_timestamps)
        self.set_temps(np.asarray(temperatures))
        self.set_pressures(np.asarray(pressures))
        self.set_solars(np.asarray(solars))
        self.set_relativehs(np.asarray(rhs))
        self.set_wind_speeds(np.asarray(wind_s))
        self.set_wind_dirs(np.asarray(wind_d))
        self.set_rain(np.asarray(rain))

    def bool_weather_set(self):
        return 0 != len(self.weather_timestamps)

    def set_spectra(self, timestamps, real_times, live_times, bin_lims, bin_cals, energys, counts_):
        self.timestamp = timestamps
        self.real_time = datetime.timedelta(seconds=real_times)
        self.live_time = datetime.timedelta(seconds=live_times)
        self.bin_lim = bin_lims
        self.bin_cal = bin_cals
        self.energy = energys
        self.counts = counts_

    def set_eff(self, eff_file):
        import spectra_utils
        self.eff_curve = np.asarray(spectra_utils.parse_eff(eff_file))

    def bool_spec_set(self):
        return 0 != (self.bin_lim[1] - self.bin_lim[0])

    def get_timestamp(self):
        return self.timestamp

    def get_weather_timestamps(self):
        return self.weather_timestamps

    def get_temps(self):
        return self.temp

    def get_pressures(self):
        return self.pressures

    def get_solars(self):
        return self.solar

    def get_relativehs(self):
        return self.relh

    def get_wind_speeds(self):
        return self.wind_speed

    def get_wind_dirs(self):
        return self.wind_dir

    def get_real_time(self):
        return self.real_time

    def get_live_time(self):
        return self.live_time

    def write_spe(self, out_file_name):
        with open(out_file_name, 'w+') as out_file:
            out_file.write(
                "$SPEC_ID:\nNo sample description was entered.\n$SPEC_REM:\nDET# 2\nDETDESC# RoofTopBEGE\nAP# GammaVision Version 6.09\n")
            out_file.write("$DATE_MEA:\n")
            tmp_date = self.timestamp
            mm_dd_yyyy = '%(mm)02d-%(dd)02d-%(yyyy)04d ' + str(tmp_date.hour) + \
                ":" + str(tmp_date.minute) + ":" + str(tmp_date.second)
            mm_dd_yyyy = mm_dd_yyyy % {"mm": tmp_date.month, "dd": tmp_date.day, "yyyy": float(
                tmp_date.year), "time": str(tmp_date.hour) + ":" + str(tmp_date.minute) + ":" + str(tmp_date.second)}
            out_file.write(mm_dd_yyyy + "\n")
            out_file.write("$MEAS_TIM:\n")
            out_file.write(str(self.live_time.total_seconds()) + " " + str(self.real_time.total_seconds()) + "\n")
            out_file.write("$DATA:\n")
            print(self.counts)
            out_file.write("1 " + str(len(self.counts)) + "\n")
            for x in range(0, len(self.counts)):
                out_file.write("{0:8}".format(int(self.counts[x])) + "\n")
            out_file.write("$ENER_FIT:\n")
            out_file.write(str(self.bin_cal[0]) + " " + str(self.bin_cal[1]))

    def write_last_update_image(self, fimage):
        from PIL import Image, ImageDraw, ImageFont

        image = Image.new("RGBA", (3000, 240), (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/home/dosenet/etc/fonts/vera_sans/Vera.ttf", 244)

        draw.text((0, 0), self.timestamp.strftime("%a, %b %d at %I:%M%p"), (0, 0, 0), font=font)
        image_resized = image.resize((3000, 240), Image.ANTIALIAS)
        image_resized.save(fimage)


class SampleCollection:
    def __init__(self):
        self.collection = []
        self.rois = []
        self.eff_curve = []

    def __str__(self):
        return str(len(self.collection))

    def add_sample(self, sample):
        self.collection.append(sample)

    def add_roi(self, file_name):
        import spectra_utils
        self.rois = spectra_utils.parse_roi(file_name)

    def set_eff(self, eff_file):
        import spectra_utils
        self.eff_curve = np.asarray(spectra_utils.parse_eff(eff_file))

    def get_eff_for_binning(self, energy_bins):
        eff = np.interp(energy_bins, self.eff_curve[:, 0], self.eff_curve[:, 1])
        return eff

    def rebin(self, delta_t):
        print("first step in rebin")
        if 0 == len(self.collection):
            print('Collection does not have any events')
            return
        k = 0
        co = []
        # print("just before loop")
        while k < len(self.collection):
            p = k
            # print("just before first timestamp call")
            tim = self.collection[p].timestamp + delta_t
            # print(tim)
            while (k < len(self.collection)) and (tim > self.collection[k].timestamp):
                if k == p:
                    k = k + 1
                    continue
                # self.collection[p].live_time = self.collection[p].live_time + self.collection[k].live_time
                # self.collection[p].real_time = self.collection[p].real_time + self.collection[k].real_time
                # self.collection[p].weather_timestamps = np.hstack([self.collection[p].weather_timestamps, self.collection[k].weather_timestamps])
                # self.collection[p].temp = np.hstack([self.collection[p].temp, self.collection[k].temp])
                # self.collection[p].pressures = np.hstack([self.collection[p].pressures, self.collection[k].pressures])
                # self.collection[p].solar = np.hstack([self.collection[p].solar, self.collection[k].solar])
                # self.collection[p].relh = np.hstack([self.collection[p].relh, self.collection[k].relh])
                # self.collection[p].wind_speed = np.hstack([self.collection[p].wind_speed, self.collection[k].wind_speed])
                # self.collection[p].wind_dir = np.hstack([self.collection[p].wind_dir, self.collection[k].wind_dir])
                # self.collection[p].counts = self.collection[p].counts + self.collection[k].counts
                # self.collection[p].rain = np.hstack([self.collection[p].rain, self.collection[k].rain])
                k = k + 1
            tmp = self.collection[p]
            tmp.live_time = np.sum([self.collection[x].live_time for x in range(p, k)])
            tmp.real_time = np.sum([self.collection[x].real_time for x in range(p, k)])
            tmp.weather_timestamps = np.asarray([y for x in range(p, k) for y in self.collection[x].weather_timestamps])
            tmp.temp = np.asarray([y for x in range(p, k) for y in self.collection[x].temp])
            tmp.pressures = np.asarray([y for x in range(p, k) for y in self.collection[x].pressures])
            tmp.solar = np.asarray([y for x in range(p, k) for y in self.collection[x].solar])
            tmp.relh = np.asarray([y for x in range(p, k) for y in self.collection[x].relh])
            tmp.wind_speed = np.asarray([y for x in range(p, k) for y in self.collection[x].wind_speed])
            tmp.wind_dir = np.asarray([y for x in range(p, k) for y in self.collection[x].wind_dir])
            if np.abs((self.collection[p].timestamp - self.collection[k - 1].timestamp).total_seconds()) > 3600:
                print("something not right here!!!", p, k,
                      self.collection[p].timestamp, self.collection[k - 1].timestamp)
            # print("Counts size: ", np.asarray(tmp.counts).shape)
            tmp.counts = np.asarray([self.collection[x].counts for x in range(p, k)])
            # print("Sum size: ", np.asarray(self.collection[p].counts).shape)
            tmp.counts = np.sum(self.collection[p].counts, axis=0)
            # print("Sumsum size:", self.collection[p].counts.shape)
            tmp.rain = np.asarray([y for x in range(p, k) for y in self.collection[x].rain])
            co.append(tmp)
        self.collection = []
        self.collection = co

    def weighted_mean(self, first, second):
        sum_ = [0, 0]
        # if np.asarray(first).shape[0] == 1 or np.asarray(first).shape[1] == 1:
        #     pass
        # else:
        #     print(np.asarray(first).shape, " ", np.asarray(second).shape)
        for n in range(0, len(first)):
            sum_[0] = sum_[0] + first[n] * second[n]
            sum_[1] = sum_[1] + first[n]
        if sum_[1] == 0 or len(first) == 0:
            return 0
        else:
            return float(sum_[0]) / float(sum_[1])

    def build_collection(self, spec_dir, weather_csv):
        import spectra_utils

        print(f"Looking for spectral files in: {spec_dir}")
        # Check if directory exists
        if not os.path.exists(spec_dir):
            print(f"ERROR: Spectral directory does not exist: {spec_dir}")

        timestamps_index = 1
        temperature_index = 2
        rh_index = 7
        pres_index = 8
        rain_index = 12
        sola_index = 13
        winds_index = 17
        windd_index = 18

        print(f"Parsing weather data from: {weather_csv}")
        list_of_lists, units, label, is_time_str, order = weather_utils.parse_weather_data(weather_csv)
        print(f"Found {len(list_of_lists[timestamps_index])} weather data points")

        file_pattern = spec_dir + '*/*.CNF'
        print(f"Searching for files matching pattern: {file_pattern}")
        fil_list = glob.glob(file_pattern)
        print(f"Found {len(fil_list)} spectral files")
        fil_list.sort()

        start_index = max(0, len(fil_list) - 10000)
        end_index = max(start_index + 1, len(fil_list) - 2)  # Ensure at least one file if possible

        print(f"Processing files {start_index} to {end_index} out of {len(fil_list)} total files")

        if start_index < len(fil_list):
            for fi in range(start_index, end_index):
                fil = fil_list[fi]
                try:
                    print(f"Processing file {fi}: {fil}")
                    k = 0
                    sa = Sample()
                    sa = spectra_utils.parse_spectra(fil, sa)

                    # Check if the sample is properly initialized
                    if not hasattr(sa, 'get_timestamp'):
                        print(f"Error: sample object not properly initialized for {fil}")
                        continue  # Skip this file

                    # Try to match weather data, but continue with empty data if none matches
                    try:
                        if len(list_of_lists[timestamps_index]) > 0:
                            p = weather_utils.discard_data_before_time(
                                list_of_lists[timestamps_index], sa.get_timestamp(), k)
                            k = weather_utils.discard_data_before_time(list_of_lists[timestamps_index],
                                                                       sa.get_timestamp() + sa.get_real_time(), p)

                            # Check if we have valid indices before accessing weather data
                            if p < k and p < len(list_of_lists[timestamps_index]) and k <= len(list_of_lists[timestamps_index]):
                                sa.set_weather_params(list_of_lists[timestamps_index][p:k - 1],
                                                      list_of_lists[temperature_index][p:k - 1],
                                                      list_of_lists[pres_index][p:k - 1],
                                                      list_of_lists[sola_index][p:k - 1],
                                                      list_of_lists[rh_index][p:k - 1],
                                                      list_of_lists[winds_index][p:k - 1],
                                                      list_of_lists[windd_index][p:k - 1],
                                                      list_of_lists[rain_index][p:k - 1])
                            else:
                                # No matching weather data - use empty arrays
                                print(f"No matching weather data for file {fil}, using empty weather data")
                                sa.set_weather_params([], [], [], [], [], [], [], [])
                        else:
                            # No weather data at all - use empty arrays
                            print(f"No weather data available for file {fil}, using empty weather data")
                            sa.set_weather_params([], [], [], [], [], [], [], [])
                    except Exception as e:
                        # Error processing weather data - use empty arrays
                        print(f"Error matching weather data for file {fil}: {e}, using empty weather data")
                        sa.set_weather_params([], [], [], [], [], [], [], [])

                    # Add the sample regardless of weather data
                    self.add_sample(sa)

                except Exception as e:
                    print(f"Error processing spectral file {fil}: {e}")
                    continue
        else:
            print("Warning: No spectral data files found matching the pattern")

        print("sample_collection::build_collection: building done")

    def write_hdf(self, file_name):
        print("sample_collection::write_hdf: starting")
        out_file = h5py.File(file_name, 'w')
        ths_yr = out_file.create_group('2014')
        timstmps = []
        specs = []
        weather_list = []
        specs_meta = []  # weather_el = np.zeros(5)
        for stmp in self.collection:
            utc_tmsmp = calendar.timegm(stmp.get_timestamp().utctimetuple())
            timstmps.append(utc_tmsmp)
            specs_meta.append([stmp.real_time.total_seconds(), stmp.live_time.total_seconds(),
                              stmp.bin_cal[0], stmp.bin_cal[1]])
            specs.append(stmp.counts)
            # print(stmp.get_timestamp())
            # if np.asarray(stmp.wind_speed).shape[0] == 1 or np.asarray(stmp.wind_speed).shape[1] == 1:
            #     pass
            # else:
            #     print(stmp.get_timestamp(), ",", np.asarray(stmp.wind_speed).shape[0], np.asarray(stmp.wind_speed).shape[1])
            if stmp.bool_weather_set():
                weather_el = [np.mean(stmp.temp), np.mean(stmp.pressures), np.mean(stmp.solar), np.mean(stmp.relh),
                              self.weighted_mean(stmp.wind_speed, stmp.wind_dir), np.mean(stmp.wind_speed), np.sum(stmp.rain)]
            else:
                weather_el = np.zeros(7)
                weather_el[:] = np.NAN
            weather_list.append(weather_el)
        ths_yr.create_dataset('timestamps', data=timstmps, dtype=np.dtype('uint32'))
        ths_yr.create_dataset('spectra', data=specs, dtype=np.dtype('uint32'))
        ths_yr.create_dataset('weather_data', data=weather_list, dtype=float)
        ths_yr.create_dataset('spectra_meta', data=specs_meta, dtype=float)
