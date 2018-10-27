# _*_ coding: utf-8 _*_
import numpy as np
import pickle
import os
import datetime

from ConvLSTM.file_reader import load_all
from operator import itemgetter
from utility.feature_processor import time_to_angle, data_coordinate_angle
from ConvLSTM.config import root, def_nan_signal


root_path = root()
nan_signal = def_nan_signal()  # 'NaN' or np.nan


# def site_to_site_no(site_pient):
#     site_pient = int(site_pient)
#     if site_pient <= 43:
#         return site_pient-1
#
#     elif site_pient >= 101:
#         return site_pient-101+44-1
#
#     else:
#         print("This site doesn't exist.")


def param_code_to_param_code_no(param_code):
    param_code = int(param_code)
    if param_code == 10:
        return 0  # 'WIND_SPEED'
    elif param_code == 11:
        return 1  # 'WIND_DIREC'
    elif param_code == 14:
        return 2  # 'AMB_TEMP'
    elif param_code == 15:
        return 3  # 'DEW_POINT'
    elif param_code == 17:
        return 4  # 'PRESSURE'
    elif param_code == 23:
        return 5  # 'RAINFALL'
    elif param_code == 38:
        return 6  # 'RH'
    elif param_code == 103:
        return 7  # 'RADIATION'
    elif param_code == 104:
        return 8  # 'SUN_HR'
    elif param_code == 105:
        return 9  # 'SUN_MIN'
    elif param_code == 106:
        return 10  # 'PRESSURE_SITE'
    elif param_code == 107:
        return 11  # 'PRESSURE_SEA'
    elif param_code == 108:
        return 12  # 'RAINFALL_HR'
    else:
        print("This param_code doesn't exist.")

EPA_feature_labels = ['SO2', 'CO', 'O3', 'PM10', 'PM2_5', 'NOx', 'NO', 'NO2', 'THC', 'NMHC', 'CH4', 'UVB', 'AMB_TEMP', 'RAINFALL', 'RH', 'WIND_SPEED', 'WIND_DIREC', 'WS_HR', 'WD_HR', 'PH_RAIN', 'RAIN_COND']
ncsist_feature_labels = ['AMB_TEMP', 'RH', 'PM2_5', 'WIND_DIREC', 'WIND_SPEED']


def pollution_to_pollution_no(pollution):
    if pollution == 'SO2':return 0
    elif pollution == 'CO':return 1
    elif pollution == 'O3':return 2
    elif pollution == 'PM10':return 3
    elif pollution == 'PM2_5':return 4
    elif pollution == 'NOx':return 5
    elif pollution == 'NO':return 6
    elif pollution == 'NO2':return 7
    elif pollution == 'THC':return 8
    elif pollution == 'NMHC':return 9
    elif pollution == 'CH4':return 10
    elif pollution == 'UVB':return 11
    elif pollution == 'AMB_TEMP':return 12
    elif pollution == 'RAINFALL':return 13
    elif pollution == 'RH':return 14
    elif pollution == 'WIND_SPEED':return 15
    elif pollution == 'WIND_DIREC':return 16
    elif pollution == 'WS_HR':return 17
    elif pollution == 'WD_HR':return 18
    elif pollution == 'PH_RAIN':return 19
    elif pollution == 'RAIN_COND':return 20
    else:
        # print("THis pollution(%s) hasn't been recorded." % pollution)
        None


"""
weather_site_name2no = {
    '台中': 20, '梧棲': 30, '大坑': 158, '梨山': 170, '思源': 173, '雙崎': 172, '馬都安': 177,
    '日月潭': 28, '合歡山莊': 39, '神木村': 143, '鳳凰': 152, '竹山': 154, '廬山': 163, '昆陽': 165, '合歡山': 241, '中興新村': 242,
    '彰師大': 36, '員林': 160, '鹿港': 164,

    '大屯山': 1, '鞍部': 5, '台北': 6, '竹子湖': 7, '信義': 221, '南港': 224, '大直': 227, '內湖': 228, '士林': 229, '大崙尾山': 230,
    '社子': 231, '石碑': 232, '天母': 233, '五指山': 234,
    '五分山': 2, '板橋': 3, '淡水': 4, '龍洞': 33, '新店': 37, '福山': 195, '福山2': 196, '福山3': 197, '桶後': 201, '大豹': 205, '四堵': 206, '屈尺': 208,
    '坪林': 209, '泰平': 212, '山佳': 213, '三貂角': 215, '永和': 216, '福隆': 219, '雙溪': 222, '大尖山': 225, '三重': 226, '鼻頭角': 235,
    '金山': 238, '三和': 239, '富貴角': 240,
    '基隆': 8,
    '拉拉山': 34, '武陵': 38, '新屋': 210, '大坪': 236,

    '蘇澳': 11, '宜蘭': 12, '南澳': 176, '太平山': 179, '東澳': 180, '東澳2': 181, '礁溪': 198, '龜山島': 200,

    '竹北': 24, '竹東': 191, '竹東2': 192, '竹東3': 193, '竹東4': 194,
    '三義': 174, '苑里': 175, '觀霧': 178, '南庄': 182, '梅花': 185, '玉蘭': 186, '羅東': 187, '羅東2': 188, '竹南': 189, '竹南2': 190,

    '花蓮': 10, '太魯閣': 35, '玉里': 137, '佳心': 139, '舞鶴': 141, '豐濱': 140, '豐濱2': 145, '光復': 149, '加路蘭山': 150, '鳳林山': 153, '水璉': 155,
    '月眉山': 156, '鯉魚潭': 159, '水源': 161, '新城': 162, '富世': 166, '富世2': 167, '大禹嶺': 168, '天祥': 169, '和中': 171, '靜浦': 244,
    '大武': 22, '成功': 26, '蘭嶼': 27, '台東': 29, '南田': 109, '大溪山': 112, '金崙': 114, '太麻里': 115, '綠島': 116, '知本': 117,
    '紅葉山': 122, '鹿野': 123, '東河': 124, '紅石': 125, '池上': 128, '向陽': 132, '長濱': 134,

    '彭佳嶼': 9,

    '金門': 13,
    '馬祖': 32, '東沙': 43,
    '東吉島': 14,
    '澎湖': 15, '吉貝': 41,

    '宜梧': 144, '草嶺': 146, '四湖': 147, '虎尾': 151, '台西': 157,
    '台南': 16, '永康': 17, '七股': 40, '善化': 127, '玉井': 129, '佳里': 130, '曾文': 131, '新營': 135, '關子嶺': 138, '南化': 243,
    '嘉義': 19, '阿里山': 21, '玉山': 23, '馬頭山': 136, '奮起湖': 142, '大埔': 148,

    '高雄': 18, '古亭坑': 120, '美濃': 121, '甲仙': 126, '表湖': 133,
    '恆春': 25, '墾丁': 42, '墾丁2': 103, '貓鼻頭': 101, '鵝鑾鼻': 102, '佳樂水': 104, '檳榔': 105, '牡丹池山': 106, '楓港': 107, '牡丹': 108,
    '琉球嶼': 110, '枋寮': 111, '潮州': 113, '三地門': 118, '尾寮山': 119
}

weather_site_no2name = {y: x for x, y in weather_site_name2no.items()}
"""


# ----------  WORK START  ----------
def data_reader(path, start_year, last_year, update=False):
    # read data from EPA files
    y_d_h_data = dict()
    not_exit_flag = 0
    while (start_year != last_year+1) and (not update):
        if os.path.exists(os.path.join(path, 'cPickle', 'pollution_and_weather_data_%s' % (str(start_year)))):
            not_exit_flag += 1
            print('Reading %d data by cPickle .. ' % start_year)
            fr = open(os.path.join(path, 'cPickle', 'pollution_and_weather_data_%s' % str(start_year)), 'rb')
            y_d_h_data[str(start_year)] = pickle.load(fr, encoding='utf-8')
            fr.close()
            start_year += 1
        else:
            break

    if not_exit_flag > 0:
        return y_d_h_data

    elif not_exit_flag == 0:
        print('Start from reading raw data.')
        # feature_vector = []
        y_d_h_data = dict()  # years, days and hours, then pollution and weather data

        # ---------- pollution ----------
        pollution_data_files = []  # multi-files
        num_pollution_property = 21

        # --- csv ---
        # csv_pollution_data = []
        load_all(pollution_data_files, os.path.join(path, 'Data_of_Air_Pollution'))

        # data pre-processing : format
        keep_date = ''
        pollution_vector_one_day = []

        for single_file_pollution_data in pollution_data_files:
            for line in single_file_pollution_data:
                if line == single_file_pollution_data[0]:
                    None
                else:
                    try:
                        if line[0].find('-') != -1:
                            line[0] = line[0].replace('-0', '/')  # 2008-01-01 -> 2008/1/1
                            line[0] = line[0].replace('-', '/')  # 2008-10-12 -> 2008/10/12
                        if line[0].find('/0') != -1:
                            line[0] = line[0].replace('/0', '/')  # 2010/01/01 ->2010/1/1
                    except:
                        print()

                    year = line[0][:line[0].find('/')]
                    date = line[0][line[0].find('/')+1:]
                    # check/create year dict., ex: 2016, 2015
                    if not(year in y_d_h_data):
                        y_d_h_data[year] = dict()
                    # check/create date dict., ex: 1/1, 10/31
                    if not(date in y_d_h_data[year]):
                        y_d_h_data[year][date] = dict()
                    # pollution sites dict.
                    if not('pollution' in y_d_h_data[year][date]):
                        y_d_h_data[year][date]['pollution'] = dict()

                    print(line[:3])

                    if keep_date != line[0]:  # a new day
                        if keep_date != '' and (keep_date[:keep_date.find('/')] == year):
                            y_d_h_data[keep_date[:keep_date.find('/')]][keep_date[keep_date.find('/')+1:]]['pollution'][line[1]] = pollution_vector_one_day
                            pollution_vector_one_day = []
                        elif keep_date != '':
                            pollution_vector_one_day = []

                        keep_date = line[0]

                        # Reserve 'num_pollution_property' entries for data, and take '-' to mean missing value
                        for each_hour in range(24):
                            pollution_vector_one_day.append(['-' for i in range(num_pollution_property)])

                    for each_hour in range(24):
                        # The first three elements are date, sites and kind of pollution
                        try:
                            pollution_vector_one_day[each_hour][pollution_to_pollution_no(line[2].replace(' ', ''))] = line[3+each_hour]
                        except:
                            break

                if line == single_file_pollution_data[-1]:  # the last recorded day of this file
                    y_d_h_data[year][date]['pollution'][line[1]] = pollution_vector_one_day

        print('--------------------------------------------------------------------------------------')

        # # ---------- weather ----------
        num_weather_property = 13
        # num_sites = 449

        weather_data = []
        load_all(weather_data, path+'Data_of_Weather/')

        for file_i in range(len(weather_data)):
            del(weather_data[file_i][0])
            # sorting by date -> site -> param_code
            for line_j in range(len(weather_data[file_i])):
                [year, _, date, angle] = time_to_angle(weather_data[file_i][line_j][2].replace(' 00:00:00', ''))

                format_day_order = angle/360.
                weather_data[file_i][line_j].append(int(year) + format_day_order)
                weather_data[file_i][line_j][0] = int(weather_data[file_i][line_j][0])

            weather_data[file_i] = sorted(weather_data[file_i], key=itemgetter(len(weather_data[file_i][line_j])-1, 0, 1))

            print('Sorted complete.')

            keep_date = ''
            keep_site = ''
            for line_j in range(len(weather_data[file_i])):
                # a new site
                if weather_data[file_i][line_j][0] != keep_site:
                    if keep_site != '':
                        site_name = keep_site  # weather_site_no2name[int(keep_site)]
                        y_d_h_data[year][date]['weather'][site_name] = weather_vector
                        print(year + '/' + date + ': site- %s' % site_name)
                    keep_site = weather_data[file_i][line_j][0]

                    weather_vector = []

                # a new day
                if weather_data[file_i][line_j][2].replace(' 00:00:00', '') != keep_date:
                    if keep_date != '' and len(weather_vector) != 0:
                        site_name = keep_site  # weather_site_no2name[int(keep_site)]
                        y_d_h_data[year][date]['weather'][site_name] = weather_vector
                        print(year + '/' + date + ': site- %s' % site_name)

                    keep_date = weather_data[file_i][line_j][2].replace(' 00:00:00', '')
                    year = keep_date[:keep_date.find('/')]
                    date = keep_date[keep_date.find('/')+1:]

                    if int(year) <= 12:  # month/date/year
                        year = keep_date[keep_date.rfind('/')+1:]
                        date = keep_date[:keep_date.rfind('/')]

                    # check/create year dict., ex: 2016, 2015
                    if not (year in y_d_h_data):
                        y_d_h_data[year] = dict()
                    # check/create date dict., ex: 1/1, 10/31
                    if not (date in y_d_h_data[year]):
                        y_d_h_data[year][date] = dict()
                    # weather sites dict.
                    if not('weather' in y_d_h_data[year][date]):
                        y_d_h_data[year][date]['weather'] = dict()

                    weather_vector = []

                # Initiate weather_vector, when 'a new day' or 'a new site'.
                if len(weather_vector) == 0:
                    for each_hour in range(24):
                        weather_vector.append(['-' for i in range(num_weather_property)])

                # collecting data
                for each_hour in range(24):
                    weather_vector[each_hour][param_code_to_param_code_no(weather_data[file_i][line_j][1])] \
                        = weather_data[file_i][line_j][3+each_hour]  # the first three element mean site, param_code and date

                if line_j == len(weather_data[file_i])-1:  # the last day
                    site_name = keep_site  # weather_site_no2name[int(keep_site)]
                    y_d_h_data[year][date]['weather'][site_name] = weather_vector
            print('----')

        print('Saving .. ')
        for years in y_d_h_data.keys():
            fw1 = open(os.path.join(path, 'cPickle', 'pollution_and_weather_data_%s' % years), 'wb')
            pickle.dump(y_d_h_data[years], fw1)
            fw1.close()

        print('Saved.')

        return y_d_h_data


def polution_data_init(table_labels):
    polution_data = dict()
    for label in table_labels:
        polution_data[label] = nan_signal
    return polution_data


def global_data_reader(dirpath, date_range=None, table_label=EPA_feature_labels):
    # read data from EPA files
    # data in each file must be sorted by date
    raw_data = list()
    load_all(raw_data, dirpath)

    polution_data_list = list()
    polution_data_index = 0

    for data_idx, data in enumerate(raw_data):
        for line_idx, line_list in enumerate(data):

            line_list[0] = line_list[0].replace('/', '-')  # year/month/day -> year-month-day
            site_name = line_list[1]
            feature = line_list[2]

            if not polution_data_index:
                print("%s  %s" % (line_list[0], site_name))
                for i in range(24):  # 24 hours in one day
                    polution_data_list.append(polution_data_init(table_label))
                polution_data_index += 1
            elif polution_data_list[-1]['time'].date() != datetime.datetime.strptime(line_list[0], "%Y-%m-%d").date():
                print("%s  %s" % (line_list[0], site_name))
                for i in range(24):  # 24 hours in one day
                    polution_data_list.append(polution_data_init(table_label))
                polution_data_index += 1

            for hour in range(24):  # 24 hours in one day
                try:
                    elem = line_list[3+hour]
                except:
                    elem = nan_signal
                insert_index = (polution_data_index-1) * 24 + hour

                polution_data_list[insert_index]['time'] = datetime.datetime.strptime(line_list[0] + "-%d" % hour, "%Y-%m-%d-%H")  # year/month/day -> year-month-day-hour
                polution_data_list[insert_index]['site'] = site_name
                try:
                    polution_data_list[insert_index][feature] = float(elem)
                except:
                    polution_data_list[insert_index][feature] = nan_signal
    return polution_data_list


def local_data_reader(dirpath, date_range=None):
    # read data from ncsist files
    filepath_list = os.listdir(dirpath)
    polution_data_list = list()

    for filename in filepath_list:

        site_name = filename.split('-')[0]

        filepath = os.path.join(dirpath, filename)
        print(filename)
        with open(filepath, 'r') as fr:
            data = fr.readlines()

        for line in data:
            try:
                # print(line)
                line_list = line.split(',')
                time_data = datetime.datetime.strptime(line_list[1], "%Y-%m-%d-%H:%M:%S")
                AMB_TEMP = float(line_list[-5])
                RH = float(line_list[-4])
                PM2_5 = float(line_list[-3])
                WIND_DIREC = float(line_list[-2])
                WIND_SPEED = float(line_list[-1])
            except:
                continue

            if not date_range:
                polution_data = dict()
                polution_data['site'] = site_name
                polution_data['time'] = time_data
                polution_data['AMB_TEMP'] = AMB_TEMP
                polution_data['RH'] = RH
                polution_data['PM2_5'] = PM2_5
                polution_data['WIND_DIREC'] = WIND_DIREC
                polution_data['WIND_SPEED'] = WIND_SPEED

                polution_data_list.append(polution_data)
            elif datetime.datetime.strptime(date_range[0], "%Y-%m-%d-%H:%M:%S") <= time_data \
                    <= datetime.datetime.strptime(date_range[1], "%Y-%m-%d-%H:%M:%S"):
                polution_data = dict()
                polution_data['site'] = site_name
                polution_data['time'] = time_data
                polution_data['AMB_TEMP'] = AMB_TEMP
                polution_data['RH'] = RH
                polution_data['PM2_5'] = PM2_5
                polution_data['WIND_DIREC'] = WIND_DIREC
                polution_data['WIND_SPEED'] = WIND_SPEED

                polution_data_list.append(polution_data)

    return polution_data_list


def db_to_dict(table):
    print("change form from db to dict ..")
    # change form of db to data form for data pre-processing(ex. y_d_h_data)
    data_dict = dict()

    # sort by time
    # sorted(table, key=lambda x: x["time"])

    for each_data in table:
        site = each_data['site']
        year = each_data['time'].year
        month = each_data['time'].month
        day = each_data['time'].day
        date = "%s/%s" % (month, day)
        hour = each_data['time'].hour
        minute = each_data['time'].minute

        if site not in data_dict:
            data_dict[site] = dict()
        if str(year) not in data_dict[site]:
            data_dict[site][str(year)] = dict()
        if date not in data_dict[site][str(year)]:
            data_dict[site][str(year)][date] = dict()
        if str(hour) not in data_dict[site][str(year)][date]:
            data_dict[site][str(year)][date][str(hour)] = dict()
        if str(minute) not in data_dict[site][str(year)][date][str(hour)]:
            data_dict[site][str(year)][date][str(hour)][str(minute)] = list()

        data_dict[site][str(year)][date][str(hour)][str(minute)].append(polution_data_init(each_data.keys()))

        for each_elem in each_data.keys():
            if each_elem in data_dict[site][str(year)][date][str(hour)][str(minute)][-1]:
                data_dict[site][str(year)][date][str(hour)][str(minute)][-1][each_elem] = each_data[each_elem]

    return data_dict


# ncsist_data_dir = "/media/clliao/006a3168-df49-4b0a-a874-891877a88870/AirQuality/dataset/PM25Data_forAI_0903-0930"
# EPA_data_dir = "/media/clliao/006a3168-df49-4b0a-a874-891877a88870/AirQuality/dataset/AirQuality_EPA/Data_of_Air_Pollution_for_testing"


# ncsist_polution_db = local_data_reader(ncsist_data_dir)
# EPA_polution_db = global_data_reader(EPA_data_dir)

# ncsist_polution_data = db_to_dict(ncsist_polution_db)
# EPA_polution_data = db_to_dict(EPA_polution_db)


# exit()
