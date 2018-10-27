# _*_ coding: utf-8 _*_
import datetime
import os
import pickle
import numpy as np
# import MySQLdb
import pymysql as MySQLdb

from utility.data_reader import data_reader, local_data_reader, global_data_reader, db_to_dict
from utility.feature_processor import time_to_angle, return_weekday, data_coordinate_angle, convert_polar_to_cartesian
from ConvLSTM.config import root, site_map2, def_nan_signal, pollution_site_map2, db_config, pollution_site_local_map
from ConvLSTM.config import def_nan_dict_global, def_nan_dict_local, def_nan_date_global, def_nan_date_local
from MySQL.connMySQL import load_db
from utility.missing_value_processer import missing_check
from utility.Utilities import topK_next_interval

root_path = root()

nan_signal = def_nan_signal()  # 'NaN' or np.nan

fake_dict_global = def_nan_dict_global()
fake_dict_date_global = def_nan_date_global()
fake_dict_local = def_nan_dict_local()
fake_dict_date_local = def_nan_date_local()

"""
num_of_pollution_data = 21
num_of_weather_data = 13

# pollution_site_map = {
#     '中部': {'台中': ['大里', '忠明', '沙鹿', '西屯', '豐原'],
#            '南投': ['南投', '竹山'],
#            '彰化': ['二林', '彰化']},
#
#     '北部': {'台北': ['中山', '古亭', '士林', '松山', '萬華'],
#            '新北': ['土城', '新店', '新莊', '板橋', '林口', '汐止', '菜寮', '萬里'],
#            '基隆': ['基隆'],
#            '桃園': ['大園', '平鎮', '桃園', '龍潭']},
#
#     '宜蘭': {'宜蘭': ['冬山', '宜蘭']},
#
#     '竹苗': {'新竹': ['新竹', '湖口', '竹東'],
#            '苗栗': ['三義', '苗栗']},
#
#     '花東': {'花蓮': ['花蓮'],
#            '台東': ['臺東']},
#
#     '北部離島': {'彭佳嶼': []},
#
#     '西部離島': {'金門': ['金門'],
#              '連江': ['馬祖'],
#              '東吉嶼': [],
#              '澎湖': ['馬公']},
#
#     '雲嘉南': {'雲林': ['崙背', '斗六'],
#             '台南': ['善化', '安南', '新營', '臺南'],
#             '嘉義': ['嘉義', '新港', '朴子']},
#
#     '高屏': {'高雄': ['仁武', '前金', '大寮', '小港', '左營', '林園', '楠梓', '美濃'],
#            '屏東': ['屏東', '恆春', '潮州']}
# }

weather_site_map = {
    '中部': {'台中': ['台中', '梧棲'],
           '南投': ['日月潭'],
           '彰化': []},

    '北部': {'台北': ['台北', '竹子湖', '鞍部', '大屯山'],
           '新北': ['板橋', '淡水'],
           '基隆': [],
           '桃園': []},

    '宜蘭': {'宜蘭': ['宜蘭', '蘇澳']},

    '竹苗': {'新竹': ['竹北'],
           '苗栗': []},

    '花東': {'花蓮': ['花蓮'],
           '台東': ['台東', '成功', '大武', '蘭嶼']},

    '北部離島': {'彭佳嶼': ['彭佳嶼']},

    '西部離島': {'金門': [],
             '連江': [],
             '東吉嶼': ['東吉島'],
             '澎湖': ['澎湖']},

    '雲嘉南': {'雲林': [],
            '台南': ['台南', '永康'],
            '嘉義': ['阿里山', '玉山', '嘉義']},

    '高屏': {'高雄': ['高雄'],
           '屏東': ['恆春']}
}

weather_site_name2no = {
    '台中': 20, '梧棲': 30, '大坑': 158, '梨山': 170, '思源': 173,
    '日月潭': 28, '合歡山莊': 39, '神木村': 143, '鳳凰': 152, '竹山': 154, '廬山': 163, '昆陽': 165, '合歡山': 241, '中興新村': 242,
    '彰師大': 36, '員林': 160, '鹿港': 164,

    '大屯山': 1, '鞍部': 5, '台北': 6, '竹子湖': 7, '信義': 221, '南港': 224, '大直': 227, '內湖': 228, '士林': 229, '大崙尾山': 230, '社子': 231, '石碑': 232, '天母': 233, '五指山': 234,
    '五分山': 2, '板橋': 3, '淡水': 4, '龍洞': 33, '新店': 37, '福山': 195, '桶後': 201, '大豹': 205, '四堵': 206, '屈尺': 208, '坪林': 209, '泰平': 212, '山佳': 213, '三貂角': 215, '永和': 216, '福隆': 219, '雙溪': 222, '大尖山': 225, '三重': 226, '鼻頭角': 235, '金山': 238, '三和': 239, '富貴角': 240,
    '基隆': 8,
    '拉拉山': 34, '武陵': 38, '新屋': 210, '大坪': 236,

    '蘇澳': 11, '宜蘭': 12, '南澳': 176, '太平山': 179, '東澳': 180, '礁溪': 198, '龜山島': 200,

    '竹北': 24, '竹東': 191,
    '三義': 174, '苑里': 175, '觀霧': 178, '南庄': 182, '梅花': 185, '玉蘭': 186, '羅東': 187, '竹南': 189,

    '花蓮': 10, '太魯閣': 35, '玉里': 137, '佳心': 139, '舞鶴': 141, '豐濱': 145, '光復': 149, '加路蘭山': 150, '鳳林山': 153, '水璉': 155, '月眉山': 156, '鯉魚潭': 159, '水源': 161, '新城': 162, '富世': 167, '大禹嶺': 168, '天祥': 169, '和中': 171, '靜浦': 244,
    '大武': 22, '成功': 26, '蘭嶼': 27, '台東': 29, '南田': 109, '大溪山': 112, '金崙': 114, '太麻里': 115, '綠島': 116, '知本': 117, '紅葉山': 122, '鹿野': 123, '東河': 124, '紅石': 125, '池上': 128, '向陽': 132, '長濱': 134,

    '彭佳嶼': 9,

    '金門': 13,
    '馬祖': 32, '東沙': 43,
    '東吉島': 14,
    '澎湖': 15, '吉貝': 41,

    '宜梧': 144, '草嶺': 146, '四湖': 147, '虎尾': 151, '台西': 157,
    '台南': 16, '永康': 17, '七股': 40, '善化': 127, '玉井': 129, '佳里': 130, '曾文': 131, '新營': 135, '關子嶺': 138, '南化': 243,
    '嘉義': 19, '阿里山': 21, '玉山': 23, '馬頭山': 136, '奮起湖': 142, '大埔': 148,

    '高雄': 18, '古亭坑': 120, '美濃': 121, '甲仙': 126, '表湖': 133,
    '恆春': 25, '墾丁': 42, '貓鼻頭': 101, '鵝鑾鼻': 102, '佳樂水': 104, '檳榔': 105, '牡丹池山': 106, '楓港': 107, '牡丹': 108, '琉球嶼': 110, '枋寮': 111, '潮州': 113, '三地門': 118, '尾寮山': 119
}

weather_site_no2name = {y: x for x, y in weather_site_name2no.items()}


def pollution_to_pollution_no_global(pollution):
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
        input("THis pollution(%s) hasn't been recorded." % pollution)
        # None


def pollution_to_pollution_no_local(pollution):
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
        input("THis pollution(%s) hasn't been recorded." % pollution)
        # None


# old version (y_d_h_data: year-month-day-site<list of hour>)

def read_global_data_map(path, site, feature_selection, date_range=[2014, 2015],
                         beginning='1/1', finish='12/31', update=False):

    # input_shape = (train_seg_length, 5, 5, feature_dim)
    y_d_h_data = data_reader(path, int(date_range[0]), int(date_range[-1]), update)

    num_of_missing = 0.
    total_number = 0.
    feature_tensor_list = []

    for year in date_range:
        print('%s .. ok' % year)
        days = 0

        for month in range(1, 13):

            # Check the exceeding of the duration
            if year == int(date_range[0]) and month < int(beginning[:beginning.index('/')]):  # start
                continue
            elif year == int(date_range[-1]) and month > int(finish[:finish.index('/')]):  # dead line
                continue

            # Set the number of days in a month
            if (month == 4) or (month == 6) or (month == 9) or (month == 11):
                days = 30
            elif month == 2:
                if '2/29' in y_d_h_data[str(year)]:
                    days = 29
                else:
                    days = 28
            else:
                days = 31

            for day in range(days):
                each_date = str(month) + '/' + str(day + 1)

                # Check the exceeding of the duration
                if (year == int(date_range[0])) and (month == int(beginning[:beginning.index('/')])) and (
                            (day+1) < int(beginning[(beginning.index('/')+1):])):  # start
                    continue
                elif (year == int(date_range[-1])) and month == int(finish[:finish.index('/')]) and (
                            (day+1) > int(finish[(finish.index('/')+1):])):  # dead line
                    continue

                if not ('pollution' in y_d_h_data[str(year)][each_date]):
                    print('Data of pollution missing: %s/%s' % (year, each_date))
                else:
                    for each_hour in range(24):

                        # Construct feature vector
                        time_feature = list()
                        time_feature += convert_polar_to_cartesian(
                            time_to_angle('%s/%s' % (year, each_date))[-1])  # day of year
                        time_feature += convert_polar_to_cartesian(
                            return_weekday(int(year), month, int(day+1)))  # day of week
                        time_feature += convert_polar_to_cartesian(
                            float(each_hour)/24*360)  # time of day

                        feature_tensor = np.zeros(shape=(site.shape + ((6 + len(feature_selection) + 1),)),
                                                  dtype=float)

                        site_names = list(site.adj_map.keys())
                        for site_name in site_names:
                            map_index = site.adj_map[site_name]

                            # Set time feature
                            feature_tensor[map_index[0], map_index[1], 0:6] = np.array(time_feature)

                            # Set feature vector
                            if not (site_name in y_d_h_data[str(year)][each_date]['pollution']):
                                # All features should be set to nan_signal
                                feature_tensor[map_index[0], map_index[1], 6:-1] = nan_signal
                                num_of_missing += len(feature_selection)
                                total_number += len(feature_selection)
                            else:
                                feature_index = 0
                                for feature_elem in feature_selection:
                                    feature_index += 1
                                    if feature_elem == 'WIND_DIREC':
                                        try:
                                            feature = float(y_d_h_data[str(year)][each_date]['pollution'][site_name][each_hour]
                                                            [pollution_to_pollution_no_global(feature_elem)])
                                            if np.isnan(feature) or feature < 0:
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index)] = nan_signal
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index+1)] = nan_signal
                                                num_of_missing += 1
                                                total_number += 1
                                            else:
                                                xy_coord = convert_polar_to_cartesian(feature)
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index)] = xy_coord[0]
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index+1)] = xy_coord[1]
                                                total_number += 1
                                        except:
                                            feature_tensor[map_index[0], map_index[1], (5+feature_index)] = nan_signal
                                            feature_tensor[map_index[0], map_index[1], (5+feature_index+1)] = nan_signal
                                            num_of_missing += 1
                                            total_number += 1
                                    elif feature_elem.find('_x_') > 0:
                                        feature_elems = feature_elem.split('_x_')
                                        try:
                                            mul_feature = 1
                                            features = np.zeros(shape=(len(feature_elems)))
                                            for i_elem in range(len(feature_elems)):
                                                features[i_elem] = float(
                                                    y_d_h_data[str(year)][each_date]['pollution'][site_name][each_hour]
                                                    [pollution_to_pollution_no_global(feature_elems[i_elem])])
                                                mul_feature *= features[i_elem]
                                            if [i for i in features if np.isnan(i)] or [i for i in features if i < 0]:
                                                feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                                                num_of_missing += 1
                                                total_number += 1
                                            else:
                                                feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = mul_feature
                                                total_number += 1
                                        except:
                                            feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                                            num_of_missing += 1
                                            total_number += 1

                                    else:
                                        try:
                                            feature = float(y_d_h_data[str(year)][each_date]['pollution'][site_name][each_hour]
                                                            [pollution_to_pollution_no_global(feature_elem)])
                                            if np.isnan(feature) or feature < 0:
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index)] = nan_signal
                                                num_of_missing += 1
                                                total_number += 1
                                            else:
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index)] = feature
                                                total_number += 1
                                        except:
                                            feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                                            num_of_missing += 1
                                            total_number += 1

                        feature_tensor_list.append(feature_tensor)

    print('Missing rate: %.5f' % (num_of_missing/total_number))
    return np.array(feature_tensor_list)
"""


def label_exist_check(feature_selection, table_label):
    pop_list = list()
    for idx, each in enumerate(feature_selection):
        if "_x_" in each:
            feature_list = each.split("_x_")
            for each_feature in feature_list:
                if each_feature not in table_label:
                    pop_list.append(idx)
                    break
        elif each not in table_label:
            print("%s isn't record in db" % each)
            pop_list.append(idx)

    pop_list.reverse()

    for idx in pop_list:
        feature_selection.pop(idx)
    print("# ----------------------------------------------------------------")
    print("#")
    print("# feature_selection: ", feature_selection)
    print("#")
    print("# ----------------------------------------------------------------")
    if not len(feature_selection):
        print("couldn't get any feature")
        exit()
    return feature_selection


# new version (y_d_h_data: site-year-month-day-hour-minute<list of second>)
# global or local, once each time
def read_global_or_local_data_map(site, feature_selection, date_range=[2014, 2015],
                         beginning='1/1', finish='12/31', table_name="ncsist_data", path='None'):
    # path: 'db' or filepath
    # load data from files
    # -------------------
    if os.path.exists(path):
        print("load files ..")
        pickle_dir = os.path.join(os.path.curdir, "..", "..", "dataset", "AirQuality_EPA", "pickle")

        pickle_exist_flag = 1
        for each_site in list(site.adj_map.keys()):
            if not os.path.exists(os.path.join(pickle_dir, '%s_db' % each_site)):
                pickle_exist_flag = 0
        if pickle_exist_flag:
            y_d_h_data = dict()
            for site_name in list(site.adj_map.keys()):
                y_d_h_data[site_name] = pickle.load(open(os.path.join(pickle_dir, '%s_db' % site_name), 'rb'))
            print()
        else:
            db_data = global_data_reader(path)
            y_d_h_data = db_to_dict(db_data)
            for site_name in y_d_h_data:
                pickle.dump(y_d_h_data[site_name], open(os.path.join(pickle_dir, '%s_db' % site_name), 'wb'))
            # exit()  # only for create pickle file of temp_db
    # -------------------

    # load data from db
    # -------------------
    # connect MySQL
    else:
        time_range = ["%d-%s-%s" % (date_range[0], beginning.split('/')[0], beginning.split('/')[1]),
                      "%d-%s-%s" % (date_range[-1], finish.split('/')[0], finish.split('/')[1])]
        time_range = [str(datetime.datetime.strptime(time_point, "%Y-%m-%d").date()) for time_point in time_range]
        print("connect db .. ")
        db = MySQLdb.connect(host=db_config["host"],
                             user=db_config["user"], passwd=db_config["passwd"], db=db_config["db"])

        polution_db = load_db(db, table_name=table_name, time_range=time_range)

        y_d_h_data = db_to_dict(polution_db)
    # -------------------
    feature_selection = label_exist_check(feature_selection, list(polution_db[0].keys()))

    num_of_missing = 0.
    total_number = 0.
    feature_tensor_list = []

    for year in date_range:
        print('%s .. ' % year)
        days = 0

        for month in range(1, 13):

            # Check the exceeding of the duration
            if year == int(date_range[0]) and month < int(beginning[:beginning.index('/')]):  # start
                continue
            elif year == int(date_range[-1]) and month > int(finish[:finish.index('/')]):  # dead line
                continue

            # Set the number of days in a month
            if (month == 4) or (month == 6) or (month == 9) or (month == 11):
                days = 30
            elif month == 2:
                # random choose two data to check whether 2/29 exist in this year
                if '2/29' in y_d_h_data[site.site_name][str(year)]:
                    days = 29
                else:
                    days = 28
            else:
                days = 31

            for day in range(days):
                each_date = str(month) + '/' + str(day + 1)

                # Check the exceeding of the duration
                if (year == int(date_range[0])) and (month == int(beginning[:beginning.index('/')])) and (
                            (day+1) < int(beginning[(beginning.index('/')+1):])):  # start
                    continue
                elif (year == int(date_range[-1])) and month == int(finish[:finish.index('/')]) and (
                            (day+1) > int(finish[(finish.index('/')+1):])):  # dead line
                    continue

                for each_hour in range(24):
                    for each_minute in range(60):
                        if each_date not in y_d_h_data[site.site_name][str(year)]:
                            num_of_missing += 24
                            total_number += 24
                            continue

                        if str(each_hour) not in y_d_h_data[site.site_name][str(year)][each_date]:
                            num_of_missing += 1
                            total_number += 1
                            continue

                        if str(each_minute) not in y_d_h_data[site.site_name][str(year)][each_date][str(each_hour)]:
                            continue

                        # Construct feature vector
                        time_feature = list()
                        time_feature += convert_polar_to_cartesian(
                            time_to_angle('%s/%s' % (year, each_date))[-1])  # day of year
                        time_feature += convert_polar_to_cartesian(
                            return_weekday(int(year), month, int(day+1)))  # day of week
                        time_feature += convert_polar_to_cartesian(
                            float(each_hour)/24*360)  # time of day

                        feature_tensor = np.zeros(shape=(site.shape + ((6 + len(feature_selection) + 1),)), dtype=float)

                        site_names = list(site.adj_map.keys())
                        for site_name in site_names:
                            map_index = site.adj_map[site_name]

                            # Set time feature
                            feature_tensor[map_index[0], map_index[1], 0:6] = np.array(time_feature)

                            # Set feature vector
                            if not (site_name in y_d_h_data):
                                # All features should be set to nan_signal
                                feature_tensor[map_index[0], map_index[1], 6:-1] = nan_signal
                                num_of_missing += len(feature_selection)
                                total_number += len(feature_selection)
                            else:
                                feature_index = 0
                                for feature_elem in feature_selection:
                                    feature_index += 1
                                    if feature_elem == 'WIND_DIREC':
                                        try:
                                            feature = float(y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
                                                            [feature_elem])
                                            if np.isnan(feature) or feature < 0:
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index)] = nan_signal
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index+1)] = nan_signal
                                                num_of_missing += 1
                                                total_number += 1
                                            else:
                                                xy_coord = convert_polar_to_cartesian(feature)
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index)] = xy_coord[0]
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index+1)] = xy_coord[1]
                                                total_number += 1
                                        except:
                                            feature_tensor[map_index[0], map_index[1], (5+feature_index)] = nan_signal
                                            feature_tensor[map_index[0], map_index[1], (5+feature_index+1)] = nan_signal
                                            num_of_missing += 1
                                            total_number += 1
                                    elif feature_elem.find('_x_') > 0:
                                        feature_elems = feature_elem.split('_x_')
                                        try:
                                            mul_feature = 1
                                            features = np.zeros(shape=(len(feature_elems)))
                                            for i_elem in range(len(feature_elems)):
                                                features[i_elem] = float(
                                                    y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
                                                    [feature_elems[i_elem]])
                                                mul_feature *= features[i_elem]
                                            if [i for i in features if np.isnan(i)] or [i for i in features if i < 0]:
                                                feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                                                num_of_missing += 1
                                                total_number += 1
                                            else:
                                                feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = mul_feature
                                                total_number += 1
                                        except:
                                            feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                                            num_of_missing += 1
                                            total_number += 1

                                    else:
                                        try:
                                            feature = float(y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
                                                            [feature_elem])
                                            if np.isnan(feature) or feature < 0:
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index)] = nan_signal
                                                num_of_missing += 1
                                                total_number += 1
                                            else:
                                                feature_tensor[map_index[0], map_index[1], (5+feature_index)] = feature
                                                total_number += 1
                                        except:
                                            feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                                            num_of_missing += 1
                                            total_number += 1

                        feature_tensor_list.append(feature_tensor)

    print('Missing rate: %.5f' % (num_of_missing/total_number))
    return np.array(feature_tensor_list)


# new version (y_d_h_data: site-year-month-day-hour-minute<list of second>)
# global and local simultaneously
# ----------------------------------------------------------------------------------------------------------------------
def create_map(year, month, day, each_hour, each_minute, site, y_d_h_data, feature_selection, num_of_missing, total_number):
    each_date = str(month) + '/' + str(day + 1)

    # Construct feature vector
    time_feature = list()
    time_feature += convert_polar_to_cartesian(
        time_to_angle('%s/%s' % (year, each_date))[-1])  # day of year
    time_feature += convert_polar_to_cartesian(
        return_weekday(int(year), month, int(day + 1)))  # day of week
    time_feature += convert_polar_to_cartesian(
        float(each_hour) / 24 * 360)  # time of day

    feature_tensor = np.zeros(shape=(site.shape + ((6 + len(feature_selection) + 1),)), dtype=float)

    site_names = list(site.adj_map.keys())
    for site_name in site_names:
        map_index = site.adj_map[site_name]

        # Set time feature
        feature_tensor[map_index[0], map_index[1], 0:6] = np.array(time_feature)

        # Set feature vector(with checking missing of whole site)
        # if not (site_name in y_d_h_data):
        #     # All features should be set to nan_signal
        #     feature_tensor[map_index[0], map_index[1], 6:-1] = nan_signal
        #     num_of_missing += len(feature_selection)
        #     total_number += len(feature_selection)
        # else:
        #     feature_index = 0
        #     for feature_elem in feature_selection:
        #         feature_index += 1
        #         if feature_elem == 'WIND_DIREC':
        #             try:
        #                 feature = float(y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
        #                                 [feature_elem])
        #                 if np.isnan(feature) or feature < 0:
        #                     feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
        #                     feature_tensor[map_index[0], map_index[1], (5 + feature_index + 1)] = nan_signal
        #                     num_of_missing += 1
        #                     total_number += 1
        #                 else:
        #                     xy_coord = convert_polar_to_cartesian(feature)
        #                     feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = xy_coord[0]
        #                     feature_tensor[map_index[0], map_index[1], (5 + feature_index + 1)] = xy_coord[1]
        #                     total_number += 1
        #             except:
        #                 feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
        #                 feature_tensor[map_index[0], map_index[1], (5 + feature_index + 1)] = nan_signal
        #                 num_of_missing += 1
        #                 total_number += 1
        #         elif feature_elem.find('_x_') > 0:
        #             feature_elems = feature_elem.split('_x_')
        #             try:
        #                 mul_feature = 1
        #                 features = np.zeros(shape=(len(feature_elems)))
        #                 for i_elem in range(len(feature_elems)):
        #                     features[i_elem] = float(
        #                         y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
        #                         [feature_elems[i_elem]])
        #                     mul_feature *= features[i_elem]
        #                 if [i for i in features if np.isnan(i)] or [i for i in features if i < 0]:
        #                     feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
        #                     num_of_missing += 1
        #                     total_number += 1
        #                 else:
        #                     feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = mul_feature
        #                     total_number += 1
        #             except:
        #                 feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
        #                 num_of_missing += 1
        #                 total_number += 1
        #
        #         else:
        #             try:
        #                 feature = float(y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
        #                                 [feature_elem])
        #                 if np.isnan(feature) or feature < 0:
        #                     feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
        #                     num_of_missing += 1
        #                     total_number += 1
        #                 else:
        #                     feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = feature
        #                     total_number += 1
        #             except:
        #                 feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
        #                 num_of_missing += 1
        #                 total_number += 1

        # Set feature vector(without checking missing of whole site)
        feature_index = 0
        for feature_elem in feature_selection:
            feature_index += 1
            ##############################################################################################################
            if feature_elem == 'WIND_DIREC':
                try:
                    feature = float(y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
                                    [feature_elem])
                    if np.isnan(feature) or feature < 0:
                        feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                        feature_tensor[map_index[0], map_index[1], (5 + feature_index + 1)] = nan_signal
                        num_of_missing += 1
                        total_number += 1
                    else:
                        xy_coord = convert_polar_to_cartesian(feature)
                        feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = xy_coord[0]
                        feature_tensor[map_index[0], map_index[1], (5 + feature_index + 1)] = xy_coord[1]
                        total_number += 1
                except:
                    feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                    feature_tensor[map_index[0], map_index[1], (5 + feature_index + 1)] = nan_signal
                    num_of_missing += 1
                    total_number += 1
            elif feature_elem.find('_x_') > 0:
                feature_elems = feature_elem.split('_x_')
                try:
                    mul_feature = 1
                    features = np.zeros(shape=(len(feature_elems)))
                    for i_elem in range(len(feature_elems)):
                        features[i_elem] = float(
                            y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
                            [feature_elems[i_elem]])
                        mul_feature *= features[i_elem]
                    if [i for i in features if np.isnan(i)] or [i for i in features if i < 0]:
                        feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                        num_of_missing += 1
                        total_number += 1
                    else:
                        feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = mul_feature
                        total_number += 1
                except:
                    feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                    num_of_missing += 1
                    total_number += 1

            else:
                try:
                    feature = float(y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)][0]
                                    [feature_elem])
                    if np.isnan(feature) or feature < 0:
                        feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                        num_of_missing += 1
                        total_number += 1
                    else:
                        feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = feature
                        total_number += 1
                except:
                    feature_tensor[map_index[0], map_index[1], (5 + feature_index)] = nan_signal
                    num_of_missing += 1
                    total_number += 1
            ##############################################################################################################

    return feature_tensor, num_of_missing, total_number


def read_hybrid_data_map(site, feature_selection, date_range=[2014, 2015],
                         beginning='1/1', finish='12/31', path='None', global_site_lock="龍潭"):
    # lock site of global to "龍潭"

    # check forms of parameter
    if len(feature_selection) != 2:
        print("error: feature_selection, feature_selection must include both EPA and ncsist, two element totally")
        exit()
    if path != 'None' and len(path) != 2:
        print("error: path, path must include both EPA and ncsist, two element totally")
        exit()

    # path: 'db' or filepath
    # path[0]: EPA
    # path[1]: ncsist
    EPA_path = path[0]
    ncsist_path = path[1]

    # feature_selection[0]: EPA
    # feature_selection[1]: ncsist
    EPA_feature_selection = feature_selection[0]
    ncsist_feature_selection = feature_selection[1]

    # load data from files
    # ------------------- un-finish: ncsist
    if os.path.exists(EPA_path):
        print("load files ..")
        pickle_dir = os.path.join(os.path.curdir, "..", "..", "dataset", "AirQuality_EPA", "pickle")

        pickle_exist_flag = 1
        for each_site in list(site.adj_map.keys()):
            if not os.path.exists(os.path.join(pickle_dir, '%s_db' % each_site)):
                pickle_exist_flag = 0
        if pickle_exist_flag:
            y_d_h_data = dict()
            for site_name in list(site.adj_map.keys()):
                y_d_h_data[site_name] = pickle.load(open(os.path.join(pickle_dir, '%s_db' % site_name), 'rb'))
            print()
        else:
            db_data = global_data_reader(path)
            y_d_h_data = db_to_dict(db_data)
            for site_name in y_d_h_data:
                pickle.dump(y_d_h_data[site_name], open(os.path.join(pickle_dir, '%s_db' % site_name), 'wb'))
            # exit()  # only for create pickle file of temp_db
    # -------------------

    # load data from db
    # -------------------
    # connect MySQL
    else:
        time_range = ["%d-%s-%s" % (date_range[0], beginning.split('/')[0], beginning.split('/')[1]),
                      "%d-%s-%s" % (date_range[-1], finish.split('/')[0], finish.split('/')[1])]
        time_range = [str(datetime.datetime.strptime(time_point, "%Y-%m-%d").date()) for time_point in time_range]

        print("connect db .. ")
        db = MySQLdb.connect(host=db_config["host"],
                             user=db_config["user"], passwd=db_config["passwd"], db=db_config["db"])

        # EPA
        table_name = "AirDataTable"
        EPA_db = load_db(db, table_name=table_name, time_range=time_range)

        if not len(EPA_db):
            print("no data in AirDataTable")
            exit()

        EPA_dict = db_to_dict(EPA_db)

        # ncsist
        table_name = "ncsist_data"
        ncsist_db = load_db(db, table_name=table_name, time_range=time_range)

        if not len(ncsist_db):
            print("no data in ncsist_data")
            exit()

        ncsist_dict = db_to_dict(ncsist_db)

        db.close()

        y_d_h_data = {**EPA_dict, **ncsist_dict}

    # -------------------
    EPA_feature_selection = label_exist_check(EPA_feature_selection, list(EPA_db[0].keys()))
    ncsist_feature_selection = label_exist_check(ncsist_feature_selection, list(ncsist_db[0].keys()))

    num_of_missing = 0.
    total_number = 0.
    fake_data_count = 0.
    EPA_feature_tensor_list = []
    ncsist_feature_tensor_list = []

    for year in date_range:
        print('%s .. ' % year)
        # days = 0

        for month in range(1, 13):

            # Check the exceeding of the duration
            if year == int(date_range[0]) and month < int(beginning[:beginning.index('/')]):  # start
                continue
            elif year == int(date_range[-1]) and month > int(finish[:finish.index('/')]):  # dead line
                continue

            # Set the number of days in a month
            if (month == 4) or (month == 6) or (month == 9) or (month == 11):
                days = 30
            elif month == 2:
                # random choose two data to check whether 2/29 exist in this year
                if '2/29' in y_d_h_data[site.site_name][str(year)]:
                    days = 29
                else:
                    days = 28
            else:
                days = 31

            for day in range(days):
                each_date = str(month) + '/' + str(day + 1)

                # Check whether time of data belong to the duration of time_range
                # -----------------------
                if (year == int(date_range[0])) and (month == int(beginning[:beginning.index('/')])) and (
                            (day+1) < int(beginning[(beginning.index('/')+1):])):  # start
                    continue
                elif (year == int(date_range[-1])) and month == int(finish[:finish.index('/')]) and (
                            (day+1) > int(finish[(finish.index('/')+1):])):  # dead line
                    continue
                # -----------------------

                for each_hour in range(24):
                    for each_minute in range(60):
                        # missing check ----------------------------------------------------------------un-finish
                        # ---------------------------
                        missing_flag = 0
                        # EPA checking
                        for site_name in site_map2()[global_site_lock].adj_map.keys():
                            if str(year) not in y_d_h_data[site_name]:
                                missing_flag = 1
                                num_of_missing += 1
                                total_number += 1
                                break

                            if each_date not in y_d_h_data[site_name][str(year)]:
                                # missing_flag = 1
                                num_of_missing += 1
                                total_number += 1
                                # create fake nan dict for following check
                                y_d_h_data[site_name][str(year)][each_date] = fake_dict_date_global
                                break

                            if str(each_hour) not in y_d_h_data[site_name][str(year)][each_date]:
                                # missing_flag = 1
                                num_of_missing += 1
                                total_number += 1
                                # create fake nan dict for following check
                                fake_data_count += 1
                                y_d_h_data[site_name][str(year)][each_date][str(each_hour)] = {'0': [fake_dict_global]}
                                break

                            # minimum unit of measurement of EPA is "hour", so only one "minute data" each hour
                            if "0" not in y_d_h_data[site_name][str(year)][each_date][str(each_hour)]:
                                # missing_flag = 1
                                num_of_missing += 1
                                total_number += 1
                                # create fake nan dict for following check
                                fake_data_count += 1
                                y_d_h_data[site_name][str(year)][each_date][str(each_hour)]['0'] = [fake_dict_global]
                                break
                        if missing_flag:
                            #############################################################################################
                            # missing data of this site this time
                            # create a site_name"NaN" data for following checking to delete
                            #############################################################################################
                            continue

                        # ncsist checking
                        for site_name in site.adj_map.keys():
                            if str(year) not in y_d_h_data[site_name]:
                                missing_flag = 1
                                num_of_missing += 1
                                total_number += 1
                                break

                            if each_date not in y_d_h_data[site_name][str(year)]:
                                # missing_flag = 1
                                num_of_missing += 1
                                total_number += 1
                                # create fake nan dict for following check
                                y_d_h_data[site_name][str(year)][each_date] = fake_dict_date_local
                                break

                            if str(each_hour) not in y_d_h_data[site_name][str(year)][each_date]:
                                # missing_flag = 1
                                num_of_missing += 1
                                total_number += 1
                                # create fake nan dict for following check
                                fake_data_count += 1
                                y_d_h_data[site_name][str(year)][each_date][str(each_hour)] = {'0': [fake_dict_local]}
                                break

                            if str(each_minute) not in y_d_h_data[site_name][str(year)][each_date][str(each_hour)]:
                                # missing_flag = 1
                                num_of_missing += 1
                                total_number += 1
                                # create fake nan dict for following check
                                fake_data_count += 1
                                y_d_h_data[site_name][str(year)][each_date][str(each_hour)][str(each_minute)] = [fake_dict_local]
                                break
                        if missing_flag:
                            #############################################################################################
                            # missing data of this site this time
                            # create a "NaN" data for following checking to delete
                            #############################################################################################
                            continue

                        total_number += 1
                        # ---------------------------

                        # EPA
                        # minimum unit of measurement of EPA is "hour", but for ncsist is minute
                        EPA_feature_tensor, num_of_missing, total_number = create_map(year, month, day, each_hour, "0", site_map2()[global_site_lock], y_d_h_data, EPA_feature_selection, num_of_missing, total_number)

                        # ncsist
                        ncsist_feature_tensor, num_of_missing, total_number = create_map(year, month, day, each_hour, each_minute, site, y_d_h_data, ncsist_feature_selection, num_of_missing, total_number)

                        EPA_feature_tensor_list.append(EPA_feature_tensor)
                        ncsist_feature_tensor_list.append(ncsist_feature_tensor)

    # print('fake_data_count: %.5f' % fake_data_count)
    # print('num_of_missing: %.5f' % num_of_missing)
    # print('total_number: %.5f' % total_number)
    print('Missing rate: %.5f' % (num_of_missing/total_number))
    return np.array(EPA_feature_tensor_list), np.array(ncsist_feature_tensor_list)
# ----------------------------------------------------------------------------------------------------------------------


# def read_local_data_map(path, site, feature_selection, date_range=["2014-1-1-00:00:00", "2015-12-31-23:59:59"], update=False):
#     all_data_list = local_data_reader(path, date_range)
#
#
#     return 0

"""
def read_data_sets(sites=['中山', '古亭', '士林', '松山', '萬華'], date_range=['2014', '2015'],
                   feature_selection=['PM2_5'], beginning='1/1', finish='12/31',
                   path=root_path+'dataset/', update=False):
    # print('Reading data .. ')

    y_d_h_data = data_reader(int(date_range[0]), int(date_range[-1]), path, update)

    # print('Reading data .. ok')
    # print('Date Range: ', date_range)
    # print('Construct feature vectors: ')
    num_of_missing = 0.
    total_number = 0.
    feature_vector_set = []
    for each_year in date_range:
        print('%s .. ok' % each_year)
        # for each_date in y_d_h_data[each_year]:
        days = 0
        for month in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']:
            # -- duration --
            if each_year == date_range[0] and int(month) < int(beginning[:beginning.index('/')]):  # start
                continue
            elif each_year == date_range[-1] and int(month) > int(finish[:finish.index('/')]):  # dead line
                continue
            # --
            if (month == '1') or (month == '3') or (month == '5') or (month == '7') or (
                        month == '8') or (month == '10') or (month == '12'):
                days = 31
            elif (month == '4') or (month == '6') or (month == '9') or (month == '11'):
                days = 30
            elif month == '2':
                if '2/29' in y_d_h_data[each_year]:
                    days = 29
                else:
                    days = 28

            for day in range(days):
                each_date = month + '/' + str(day + 1)

                # -- duration --
                if (each_year == date_range[0]) and (int(month) == int(beginning[:beginning.index('/')])) and (
                            (day+1) < int(beginning[(beginning.index('/')+1):])):  # start
                    continue
                elif (each_year == date_range[-1]) and int(month) == int(finish[:finish.index('/')]) and (
                            (day+1) > int(finish[(finish.index('/')+1):])):  # dead line
                    continue
                # --

                if not ('pollution' in y_d_h_data[each_year][each_date]):
                    print('Data of pollution missing: %s/%s' % (each_year, each_date))
                else:
                    for each_hour in range(24):
                        feature_vector = list()
                        feature_vector += data_coordinate_angle(
                            time_to_angle('%s/%s' % (each_year, each_date))[-1])  # day of year
                        feature_vector += data_coordinate_angle(
                            return_weekday(int(each_year), int(month), int(day+1)))  # day of week
                        feature_vector += data_coordinate_angle(
                            float(each_hour)/24*360)  # time of day
                        for site in sites:
                            if not (site in y_d_h_data[each_year][each_date]['pollution']):
                                # print('Data of site(%s) missing: %s/%s %d:00' % (site, each_year, each_date, each_hour))
                                for feature_elem in feature_selection:
                                    feature_vector.append(nan_signal)
                                    num_of_missing += 1
                                    total_number += 1
                            else:
                                for feature_elem in feature_selection:
                                    try:
                                        feature = float(y_d_h_data[each_year][each_date]['pollution'][site][each_hour][pollution_to_pollution_no_global(feature_elem)])
                                        if feature < 0:
                                            feature_vector.append(nan_signal)
                                            num_of_missing += 1
                                            total_number += 1
                                        else:
                                            feature_vector.append(feature)
                                            total_number += 1
                                    except:
                                        # print('Data of feature(%s) of site(%s) missing: %s/%s %d:00' % (
                                        #     feature_elem, site, each_year, each_date, each_hour))
                                        feature_vector.append(nan_signal)
                                        num_of_missing += 1
                                        total_number += 1
                        feature_vector_set.append(feature_vector)

    # print('data_frame .. ok')
    print('Missing rate: %.5f' % (num_of_missing/total_number))
    return feature_vector_set
"""


def concatenate_time_steps(X, n_steps):
    # input X should be a 2d-array
    # output Y will be a 3d-array
    # remainder element will be ignored
    # y are the elements of Y
    length = len(X)
    Y = []
    for i in range(length):
        y = []
        if (i + n_steps) <= length:
            for j in range(n_steps):
                y += list(X[i+j])
            Y.append(y)
    return Y


def construct_time_map(X, n_steps):
    # X: input 4d-tensor
    # n_steps: train_seg_length
    # Y: output 5d-tensor
    # y are the elements of Y
    length = len(X) - n_steps
    Y = []
    for i in range(length):
        y = []
        for j in range(n_steps):
            y.append(X[i+j])
        Y.append(y)
    return np.array(Y)


def construct_time_map2(X, train_seq_seg):
    # create time series
    # X: input 4d-tensor
    # train_seq_seg: e.g., [(6, 1), (24, 2), (48, 3), (96, 6), (192, 12)]
    # Y: output 5d-tensor
    # y are the elements of Y
    length = len(X) - train_seq_seg[-1][0]
    Y = []
    for i in range(length):
        y = []
        for j in range(len(train_seq_seg) - 1):
            seg_idx = len(train_seq_seg) - j - 1
            sum_for_avg = 0
            for idx in range(train_seq_seg[seg_idx][0], train_seq_seg[seg_idx-1][0], -1):
                sum_for_avg += X[i + train_seq_seg[-1][0] - idx]
                if (idx-1) % train_seq_seg[seg_idx][1] == 0:
                    y.append(sum_for_avg / train_seq_seg[seg_idx][1])
                    sum_for_avg = 0
        sum_for_avg = 0
        for idx in range(train_seq_seg[0][0], 0, -1):
            sum_for_avg += X[i + train_seq_seg[-1][0] - idx]
            if (idx-1) % train_seq_seg[0][1] == 0:
                y.append(sum_for_avg / train_seq_seg[0][1])
                sum_for_avg = 0
        Y.append(y)
    return np.array(Y)


def construct_time_map_with_label(X, label, train_seq_seg, time_unit=1):
    # create time series
    # X: input 4d-tensor
    # train_seq_seg: e.g., [(6, 1), (24, 2), (48, 3), (96, 6), (192, 12)]
    # Y: output 5d-tensor
    # y are the elements of Y
    length = len(X) - train_seq_seg[-1][0]
    Y = []
    new_label = []
    for i in range(0, length, time_unit):
        y = []
        for j in range(len(train_seq_seg) - 1):
            seg_idx = len(train_seq_seg) - j - 1
            sum_for_avg = 0
            for idx in range(train_seq_seg[seg_idx][0], train_seq_seg[seg_idx-1][0], -1):
                sum_for_avg += X[i + train_seq_seg[-1][0] - idx]
                if (idx-1) % train_seq_seg[seg_idx][1] == 0:
                    y.append(sum_for_avg / train_seq_seg[seg_idx][1])
                    sum_for_avg = 0
        sum_for_avg = 0
        # sum_for_avg_label = 0
        for idx in range(train_seq_seg[0][0], 0, -1):
            sum_for_avg += X[i + train_seq_seg[-1][0] - idx]
            if (idx-1) % train_seq_seg[0][1] == 0:
                y.append(sum_for_avg / train_seq_seg[0][1])
                sum_for_avg = 0
        Y.append(y)
        # get label
        # new_label.append(label[i])
        new_label.append(topK_next_interval(label[i:i+train_seq_seg[0][1]], train_seq_seg[0][1], 10)[0])
    return np.array(Y), np.array(new_label)


def construct_time_steps(X, n_steps):
    # input X should be a 2d-array
    # output Y will be a 3d-array
    # y are the elements of Y
    length = len(X)
    Y = []
    for i in range(length):
        y = []
        if (i + n_steps) <= length:
            for j in range(n_steps):
                y.append(X[i+j])
            Y.append(y)
    return Y


def construct_second_time_steps(X, n_steps_layer1, n_steps_layer2):
    length = len(X)
    Y = list()
    for i in range(length):
        y = list()
        if (i + n_steps_layer1 * n_steps_layer2) <= length:
            for j in range(n_steps_layer2):
                y.append(X[i + j*n_steps_layer1])
            Y.append(y)
    return Y


# if __name__ == "__main__":
#     # from utility.reader import read_hybrid_data_map
#
#     # --------- EPA testing
#     """
#     EPA_data_dir = "/media/clliao/006a3168-df49-4b0a-a874-891877a88870/AirQuality/dataset/AirQuality_EPA/Data_of_Air_Pollution"
#
#     training_year = [2014, 2015]  # change format from   2014-2015   to   ['2014', '2015']
#     training_duration = ['1/1', '12/31']
#
#     if training_year[0] == training_year[-1]:
#         training_year.pop(0)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # kinds of pollution in AirQuality EPA DB:
#     #  'PSI', 'MajorPollutant', 'So2', 'CO', 'O3', 'PM10', 'PM2_5', 'NO2', 'WindSpeed', 'WindDirec', 'FPMI', 'NOx', 'NO'
#     # ------------------------------------------------------------------------------------------------------------------
#     pollution_kind = ['PM2_5', 'O3', 'SO2', 'CO', 'NOx', 'NO', 'NO2', 'AMB_TEMP', 'RH',
#                       'PM2_5_x_O3', 'PM2_5_x_CO', 'PM2_5_x_NOx', 'O3_x_CO', 'O3_x_NOx', 'O3_x_AMB_TEMP', 'CO_x_NOx',
#                       'WIND_SPEED', 'WIND_DIREC']
#
#     target_site = pollution_site_map2["古亭"]
#
#     a = read_global_or_local_data_map(site=target_site, feature_selection=pollution_kind,
#                                       date_range=np.atleast_1d(training_year), beginning=training_duration[0],
#                                       finish=training_duration[-1], table_name="AirDataTable")
#     """
#     # --------- ncsist testing
#     """
#     # ncsist_data_dir = "/media/clliao/006a3168-df49-4b0a-a874-891877a88870/AirQuality/dataset/PM25Data_forAI_0903-0930"
#     training_year = [2018, 2018]  # change format from   2014-2015   to   ['2014', '2015']
#     training_duration = ['9/1', '9/30']
#
#     if training_year[0] == training_year[-1]:
#         training_year.pop(0)
#
#     # ------------------------------------------------------------------------------------------------------------------
#     # kinds of pollution in AirQuality ncsist DB:
#     #  'AMB_TEMP', 'PM2_5', 'RH', 'WIND_SPEED', 'WIND_DIREC'
#     # ------------------------------------------------------------------------------------------------------------------
#     pollution_kind = ['AMB_TEMP', 'PM2_5', 'RH', 'WIND_SPEED', 'WIND_DIREC']
#
#     target_site = pollution_site_local_map["Node_09"]
#
#     a = read_global_or_local_data_map(site=target_site, feature_selection=pollution_kind,
#                                       date_range=np.atleast_1d(training_year), beginning=training_duration[0],
#                                       finish=training_duration[-1], table_name="ncsist_data")
#      """
#
#
#     # --------- hybrid testing
#     training_year = [2018, 2018]  # change format from   2014-2015   to   ['2014', '2015']
#     training_duration = ['9/2', '9/5']
#     # training_duration = ['9/1', '9/30']
#
#     if training_year[0] == training_year[-1]:
#         training_year.pop(0)
#
#     pollution_kind = [
#         # EPA
#         ['PM2_5', 'O3', 'SO2', 'CO', 'NOx', 'NO', 'NO2', 'AMB_TEMP', 'RH',
#          'PM2_5_x_O3', 'PM2_5_x_CO', 'PM2_5_x_NOx', 'O3_x_CO', 'O3_x_NOx', 'O3_x_AMB_TEMP', 'CO_x_NOx',
#          'WIND_SPEED', 'WIND_DIREC'],
#         # ncsist
#         ['AMB_TEMP', 'PM2_5', 'RH', 'WIND_SPEED', 'WIND_DIREC']
#     ]
#     target_site = pollution_site_local_map["Node_09"]
#     EPA_feature_tensor_list, ncsist_feature_tensor_list = read_hybrid_data_map(site=target_site,
#                                                                                feature_selection=pollution_kind,
#                                                                                date_range=np.atleast_1d(training_year),
#                                                                                beginning=training_duration[0],
#                                                                                finish=training_duration[-1],
#                                                                                global_site_lock="龍潭")
#     EPA_feature_tensor_list = missing_check(EPA_feature_tensor_list)
#     EPA_feature_tensor_list = construct_time_map2(EPA_feature_tensor_list, [(2*60, 1*60)])
#     exit()
