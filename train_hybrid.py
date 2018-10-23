import time
import json

# from ConvLSTM.global_forecast_model import GlobalForecastModel
# from ConvLSTM.hybrid_forecast_model import HybridForecastModel, GlobalForecastModel
from ConvLSTM.hybrid_forecast_model import HybridForecastModel
from ConvLSTM.config import *

from utility.reader import read_hybrid_data_map, construct_time_map2, read_global_or_local_data_map, \
    construct_time_map_with_label
from utility.missing_value_processer import missing_check
from utility.Utilities import *
from ConvLSTM.config import root, dataset_path, site_map2, pollution_site_local_map, global_site_lock

#
# ----- START: Parameters Declaration ----------------------------------------------------------------------------------
#

# Define data loading parameters

# Notice that 'WIND_SPEED' must be the last feature and 'WIND_DIREC' must be the last second.
# pollution_kind = ['PM2_5', 'O3', 'SO2', 'CO', 'NOx', 'NO', 'NO2', 'AMB_TEMP', 'RH',
#                   'PM2_5_x_O3', 'PM2_5_x_CO', 'PM2_5_x_NOx', 'O3_x_CO', 'O3_x_NOx', 'O3_x_AMB_TEMP', 'CO_x_NOx',
#                   'WIND_SPEED', 'WIND_DIREC']

pollution_kind = [
        # EPA
        ['PM2_5', 'O3', 'SO2', 'CO', 'NOx', 'NO', 'NO2', 'AMB_TEMP', 'RH',
         'PM2_5_x_O3', 'PM2_5_x_CO', 'PM2_5_x_NOx', 'O3_x_CO', 'O3_x_NOx', 'O3_x_AMB_TEMP', 'CO_x_NOx',
         'WIND_SPEED', 'WIND_DIREC'],
        # ncsist
        ['AMB_TEMP', 'PM2_5', 'RH', 'WIND_SPEED', 'WIND_DIREC']
    ]

# 'day of year', 'day of week' and 'time of day' respectively are represented by two dimensions
global_feature_kind_shift = 6
local_feature_kind_shift = 6

global_pollution_kind = pollution_kind[0]  # EPA
local_pollution_kind = pollution_kind[1]  # ncsist

# target_kind = ['PM2_5', 'O3', 'NO', 'NO2', 'NOx']
local_target_kind = ['PM2_5']
global_target_kind = local_target_kind  # due to hybrid_model doesn't do any processing of output_form of local and global, target_kind for global and local must be the same

# Define target duration for training
training_year = [2018, 2018]  # change format from   2014-2015   to   ['2014', '2015']
# training_duration = ['9/3', '9/5']
training_duration = ['9/1', '9/20']

# Define target duration for prediction
testing_year = [2018, 2018]
# testing_duration = ['9/5', '9/8']
testing_duration = ['9/21', '9/30']

model_root_path = '/media/clliao/006a3168-df49-4b0a-a874-891877a888701/AirQuality/PM25Forecast/ConvLSTM/models'
# Using model params to constract model's name if model_name == '' or None
model_name = ''

##########################################################################################################################
# predict the target of the subsequent hours as the label
# Note: change time unit from hour to minute !!!!!!!!!!!!!!!!!!!!!!!!
interval_hours = 1 #* 60
##########################################################################################################################

# Define model parameters
###############################################################################################################################
global_hyper_params = {
        'num_filters': 16,
        'kernel_size': (3, 3),
        'regularizer': 1e-7,
        'cnn_dropout': 0.5,
        'r_dropout': 0.5,
        'pool_size': (2, 2),
        '3d_pool_size': (1, 5, 5),
        'epoch': 100,
        'batch_size': 256,
        'interval_hours': interval_hours
    }
local_hyper_params = {
        'num_filters': 16,
        'kernel_size': (3, 3),
        'regularizer': 1e-7,
        'cnn_dropout': 0.5,
        'r_dropout': 0.5,
        'pool_size': (2, 2),
        '3d_pool_size': (2, 2, 2),
        'epoch': 100,
        'batch_size': 256,
        'interval_hours': interval_hours
    }
###############################################################################################################################

valid_ratio = 0.1

# Define the output form
local_output_form = (0, 80, 1, len(local_target_kind))      # Regression (min, max, num_of_slice, num_of_obj)
global_output_form = (0, 80, 1, len(global_target_kind))      # Regression (min, max, num_of_slice, num_of_obj)

# output_form = (0, 150, 50, 1)                 # Classification (min, max, num_of_slice, num_of_obj)
# if local_output_form[2] > 1:
#     local_mean_form = [i for i in range(local_output_form[0], local_output_form[1], int(local_output_form[1]/local_output_form[2]))]
#     local_mean_form = [i+int(local_output_form[1]/local_output_form[2])/2 for i in mean_form]
# if global_output_form[2] > 1:
#     local_mean_form = [i for i in range(global_output_form[0], global_output_form[1], int(global_output_form[1]/global_output_form[2]))]
#     local_mean_form = [i+int(global_output_form[1]/global_output_form[2])/2 for i in mean_form]
# output_size = output_form[2] * output_form[3]

# Define target site and its adjacent map for prediction
pollution_site_map2 = site_map2()

for target_site_keys in pollution_site_local_map.keys():
    # print(pollution_site_map2[target_site_keys].site_name)
    # global
    global_site = pollution_site_map2[global_site_lock]  # 龍潭
    global_center_i = int(global_site.shape[0] / 2)
    global_center_j = int(global_site.shape[1] / 2)

    # local
    target_site = pollution_site_local_map[target_site_keys]
    center_i = int(target_site.shape[0]/2)
    center_j = int(target_site.shape[1]/2)
    local = target_site.local
    city = target_site.city
    target_site_name = target_site.site_name
    site_list = list(target_site.adj_map.keys())  # ['士林', '中山', '松山', '汐止', '板橋', '萬華', '古亭', '土城', '新店']
    map_shape = target_site.shape

    # Define pre-processing parameters

    #################################################################################################################################
    global_train_seq_seg = [(24*60, 60)]   # [(12, 1), (24, 2), (48, 3), (72, 6)]
    local_train_seq_seg = [(24*60, 60)]
    #################################################################################################################################
    train_seq_length = int(global_train_seq_seg[0][0] / global_train_seq_seg[0][1])
    for seg_idx in range(1, len(global_train_seq_seg)):
        train_seq_length += int((global_train_seq_seg[seg_idx][0] - global_train_seq_seg[seg_idx-1][0]) / global_train_seq_seg[seg_idx][1])
    #
    # ----- END: Parameters Declaration --------------------------------------------------------------------------------
    #


    #
    # ----- START: Year Processing -------------------------------------------------------------------------------------
    #

    # Clear redundant year, i.e., [2014, 2014] ==> [2014]
    if training_year[0] == training_year[-1]:
        training_year.pop(0)
    if testing_year[0] == testing_year[-1]:
        testing_year.pop(0)

    # Generate years sequence, i.e., [2014, 2016] ==> [2014, 2015, 2016]
    range_of_year = training_year[-1] - training_year[0]
    for i in range(range_of_year):
        if not(int(i + training_year[0]) in training_year):
            training_year.insert(i, int(i + training_year[0]))

    #
    # ----- END: Year Processing ---------------------------------------------------------------------------------------
    #


    #
    # ----- START: Data Loading ----------------------------------------------------------------------------------------
    #

    # Set the path of training & testing data
    root_path = root()
    data_path = dataset_path()
    testing_month = testing_duration[0][:testing_duration[0].index('/')]
    folder = root_path
    training_start_point = training_duration[0][:training_duration[0].index('/')]
    training_end_point = training_duration[-1][:training_duration[-1].index('/')]
    print('site: %s' % target_site_name)
    print('Training for %s/%s to %s/%s' % (str(training_year[0]), training_duration[0],
                                           str(training_year[-1]), training_duration[-1]))
    print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))
    print('global target: %s' % global_target_kind)
    print('local target: %s' % local_target_kind)

    # Set start time of data loading.
    print('Loading data .. ')
    start_time = time.time()
    initial_time = time.time()

    # Load training data, where: size(X_train) = (data_size, map_l, map_w, map_h), not sequentialized yet.
    print('Preparing training dataset ..')
    X_train_global, X_train_local = read_hybrid_data_map(site=target_site, feature_selection=pollution_kind,
                                   date_range=np.atleast_1d(training_year), beginning=training_duration[0],
                                   finish=training_duration[-1])

    # ============================= START: FOR training global model ===========================================================
    # full_training_year = [2017, 2018]  # change format from   2014-2015   to   ['2014', '2015']
    # full_training_duration = ['12/1', '8/31']
    full_training_year = [2017, 2018]  # change format from   2014-2015   to   ['2014', '2015']
    full_training_duration = ['9/1', '8/30']

    if full_training_year[0] == full_training_year[-1]:
        full_training_year.pop(0)
    full_pollution_kind = ['PM2_5', 'O3', 'SO2', 'CO', 'NOx', 'NO', 'NO2', 'AMB_TEMP', 'RH',
                          'PM2_5_x_O3', 'PM2_5_x_CO', 'PM2_5_x_NOx', 'O3_x_CO', 'O3_x_NOx', 'O3_x_AMB_TEMP', 'CO_x_NOx',
                          'WIND_SPEED', 'WIND_DIREC']

    full_target_site = pollution_site_map2["龍潭"]
    global_full_seq_seg = [(24, 1)]
    # global_full_seq_seg = global_train_seq_seg

    X_train_global_full = read_global_or_local_data_map(site=full_target_site, feature_selection=full_pollution_kind,
                                                        date_range=np.atleast_1d(full_training_year),
                                                        beginning=full_training_duration[0],
                                                        finish=full_training_duration[-1], table_name="AirDataTable")

    print("missing checking(global_FULL) ..")
    X_train_global_full = missing_check(X_train_global_full)
    print("missing checking(global_FULL) ..")
    Y_train_global_full = np.array(X_train_global_full)[:, global_center_i, global_center_j,
                          [global_feature_kind_shift + global_pollution_kind.index(i) for i in global_target_kind]]
    # X_train_global_full = construct_time_map2(X_train_global_full[:-1], global_full_seq_seg)

    X_train_global_full, Y_train_global_full = construct_time_map_with_label(X_train_global_full[:-1],
                                                                             Y_train_global_full[:-1],
                                                                             global_full_seq_seg)

    remain_list = []
    for i in range(len(X_train_global_full)):
        # Check missing or not, if X_train_global[i] is missing, then this command will return True
        if not np.isnan(np.sum(X_train_global_full[i])):
            remain_list.append(i)
    X_train_global_full = X_train_global_full[remain_list]
    Y_train_global_full = Y_train_global_full[remain_list]

    mean_y_train_global_full = np.zeros(shape=(global_output_form[3],))
    std_y_train_global_full = np.zeros(shape=(global_output_form[3],))
    if global_output_form[2] == 1:
        for i in range(global_output_form[3]):
            print('Regression: Normalize Y_train_global[%d] ..' % i)
            mean_y_train_global_full[i] = np.mean(Y_train_global_full[:, i])
            std_y_train_global_full[i] = np.std(Y_train_global_full[:, i])

            if not std_y_train_global_full[i]:
                input("Denominator cannot be 0.")
            Y_train_global_full[:, i] = np.array(
                [(y - mean_y_train_global_full[i]) / std_y_train_global_full[i] for y in Y_train_global_full[:, i]])
            print(
                'mean_y_train_FULL_global: %f  std_y_train_FULL_global: %f' % (mean_y_train_global_full[i],
                                                                               std_y_train_global_full[i]))

    # Construct corresponding label.
    Y_train_global_full = np.array(
        [Y_train_global_full[int(global_full_seq_seg[-1][0]/global_full_seq_seg[-1][1]):, i] for i in range(global_output_form[3])]).transpose()
    # Y_train_global_full = np.array(
    #     [topK_next_interval(Y_train_global_full[:, i], interval_hours, 1) for i in range(global_output_form[3])]).transpose()
    #     # [topK_next_interval(Y_train_global_full[:, i], interval_hours, 1) for i in range(global_output_form[3])]).transpose()

    # Make
    # ############### Using len() instead of size here #####################
    batch = np.min([len(Y_train_global_full), len(X_train_global_full)])
    X_train_global_full = X_train_global_full[:batch]
    Y_train_global_full = Y_train_global_full[:batch]

    # Validation set
    num_split_to_valid_f = int(len(X_train_global_full) * valid_ratio)
    X_valid_global_full = X_train_global_full[-num_split_to_valid_f:]
    Y_valid_global_full = Y_train_global_full[-num_split_to_valid_f:]

    # Training set
    X_train_global_full = X_train_global_full[:-num_split_to_valid_f]
    Y_train_global_full = Y_train_global_full[:-num_split_to_valid_f]

    # Generate model_input3 from model_input (LSTM)
    X_train_global3_full = X_train_global_full[:, :, global_center_i, global_center_j, :]
    X_valid_global3_full = X_valid_global_full[:, :, global_center_i, global_center_j, :]
    # X_test_global3 = X_test_global[:, :, global_center_i, global_center_j, :]
    # ============================= END: FOR training global model ===========================================================

    print("missing checking(global) ..")
    X_train_global = missing_check(X_train_global)
    print("missing checking(local) ..")
    X_train_local = missing_check(X_train_local)
    Y_train_global = np.array(X_train_global)[:, global_center_i, global_center_j, [global_feature_kind_shift + global_pollution_kind.index(i) for i in global_target_kind]]
    Y_train = np.array(X_train_local)[:, center_i, center_j, [local_feature_kind_shift + local_pollution_kind.index(i) for i in local_target_kind]] # local

    # Load testing data, where: size(X_test) = (data_size, map_l, map_w, map_h), not sequentialized yet.
    print('Preparing testing dataset ..')
    X_test_global, X_test_local = read_hybrid_data_map(site=target_site, feature_selection=pollution_kind,
                                  date_range=np.atleast_1d(testing_year), beginning=testing_duration[0],
                                  finish=testing_duration[-1])
    X_test_global = missing_check(X_test_global)
    X_test_local = missing_check(X_test_local)
    Y_test_global = np.array(X_test_global)[:, global_center_i, global_center_j, [local_feature_kind_shift + global_pollution_kind.index(i) for i in global_target_kind]]
    Y_test = np.array(X_test_local)[:, center_i, center_j, [local_feature_kind_shift + local_pollution_kind.index(i) for i in local_target_kind]] # local

    # Set end time of data loading
    final_time = time.time()
    print('Reading data .. ok, ', end='')
    time_spent_printer(start_time, final_time)

    #
    # ----- END: Data Loading ------------------------------------------------------------------------------------------
    #


    #
    # ----- START: Data Pre-processing ---------------------------------------------------------------------------------
    #

    # Construct sequential data.
    print('Construct time series dataset ..')
    start_time = time.time()
    ######################################################################################################################
    # construct time seq without y so label length will >> training data length

    # X_train_global = construct_time_map2(X_train_global[:-1], global_train_seq_seg)
    # X_train_local = construct_time_map2(X_train_local[:-1], local_train_seq_seg)
    # X_test_global = construct_time_map2(X_test_global[:-1], global_train_seq_seg)
    # X_test_local = construct_time_map2(X_test_local[:-1], local_train_seq_seg)
    ######################################################################################################################
    X_train_global, Y_train_global = construct_time_map_with_label(X_train_global[:-1],
                                                                   Y_train_global[:-1], global_train_seq_seg)
    X_train_local, Y_train = construct_time_map_with_label(X_train_local[:-1],
                                                           Y_train[:-1], local_train_seq_seg)
    X_test_global, Y_test_global = construct_time_map_with_label(X_test_global[:-1],
                                                                 Y_test_global[:-1], global_train_seq_seg)
    X_test_local, Y_test = construct_time_map_with_label(X_test_local[:-1],
                                                         Y_test[:-1], local_train_seq_seg)
    final_time = time.time()
    time_spent_printer(start_time, final_time)

    ##############################################################################################################################################################
    # pop missing/NaN of data
    # form: ConvLstm
    # must delete X and Y simultaneously
    print('pop missing training data')
    start_time = time.time()
    remain_list = []
    for i in range(len(X_train_global)):
        # Check missing or not, if X_train_global[i] is missing, then this command will return True
        if not np.isnan(np.sum(X_train_global[i])):
            remain_list.append(i)
    X_train_global = X_train_global[remain_list]
    Y_train_global = Y_train_global[remain_list]
    X_train_local = X_train_local[remain_list]
    Y_train = Y_train[remain_list]
    time_spent_printer(start_time, time.time())
    # global
    remain_list = []
    for i in range(len(X_train_local)):
        # Check missing or not, if X_train_local[i] is missing, then this command will return True
        if not np.isnan(np.sum(X_train_local[i])):
            remain_list.append(i)
    X_train_global = X_train_global[remain_list]
    Y_train_global = Y_train_global[remain_list]
    X_train_local = X_train_local[remain_list]
    Y_train = Y_train[remain_list]
    time_spent_printer(start_time, time.time())
    ##############################################################################################################################################################

    # Regression: Normalize the dependent variable Y in the training dataset.
    # mean_y_train = np.zeros(shape=(local_output_form[3],))
    # std_y_train = np.zeros(shape=(local_output_form[3],))
    #
    # mean_y_train_global = np.zeros(shape=(global_output_form[3],))
    # std_y_train_global = np.zeros(shape=(global_output_form[3],))

    # Regression: Normalize the dependent variable Y in the history EPA dataset.
    mean_y_train = mean_y_train_global_full
    std_y_train = std_y_train_global_full

    mean_y_train_global = mean_y_train_global_full
    std_y_train_global = std_y_train_global_full
    if local_output_form[2] == 1:
        for i in range(local_output_form[3]):
            print('Regression: Normalize Y_train[%d] (Y_train_local) ..' % i)

            # mean_y_train[i] = np.mean(Y_train[:, i])
            # std_y_train[i] = np.std(Y_train[:, i])

            if not std_y_train[i]:
                input("Denominator cannot be 0.")
            Y_train[:, i] = np.array([(y - mean_y_train[i]) / std_y_train[i] for y in Y_train[:, i]])
            print('mean_y_train: %f  std_y_train: %f' % (mean_y_train[i], std_y_train[i]))

    if global_output_form[2] == 1:
        for i in range(global_output_form[3]):
            print('Regression: Normalize Y_train_global[%d] ..' % i)
            # mean_y_train_global[i] = np.mean(Y_train_global[:, i])
            # std_y_train_global[i] = np.std(Y_train_global[:, i])

            if not std_y_train_global[i]:
                input("Denominator cannot be 0.")
            Y_train_global[:, i] = np.array(
                [(y - mean_y_train_global[i]) / std_y_train_global[i] for y in Y_train_global[:, i]])
            print(
                'mean_y_train_global: %f  std_y_train_global: %f' % (mean_y_train_global[i], std_y_train_global[i]))


    # ###############################################################################################################################################
    # -- For fit construct_time_map_with_label (time unit is one hour)
    # Construct corresponding label.
    Y_train = np.array([Y_train[int(local_train_seq_seg[-1][0]/local_train_seq_seg[-1][-1]):, i] for i in range(local_output_form[3])]).transpose()
    # Y_train = np.array([topK_next_interval(Y_train[:, i], interval_hours, 1) for i in range(local_output_form[3])]).transpose()
    Y_train_global = np.array([Y_train_global[int(global_train_seq_seg[-1][0]/global_train_seq_seg[-1][-1]):, i] for i in range(global_output_form[3])]).transpose()
    # Y_train_global = np.array([topK_next_interval(Y_train_global[:, i], interval_hours, 1) for i in range(global_output_form[3])]).transpose()

    Y_test = np.array([Y_test[int(local_train_seq_seg[-1][0]/local_train_seq_seg[-1][-1]):, i] for i in range(local_output_form[3])]).transpose()
    # Y_test = np.array([topK_next_interval(Y_test[:, i], interval_hours, 1) for i in range(local_output_form[3])]).transpose()
    Y_test_global = np.array([Y_test_global[int(global_train_seq_seg[-1][0]/global_train_seq_seg[-1][-1]):, i] for i in range(global_output_form[3])]).transpose()
    # Y_test_global = np.array([topK_next_interval(Y_test_global[:, i], interval_hours, 1) for i in range(global_output_form[3])]).transpose()

    # # -- old --
    # # Construct corresponding label.
    # Y_train = np.array([Y_train[local_train_seq_seg[-1][0]:, i] for i in range(local_output_form[3])]).transpose()
    # Y_train = np.array([topK_next_interval(Y_train[:, i], interval_hours, 1) for i in range(local_output_form[3])]).transpose()
    # Y_train_global = np.array([Y_train_global[global_train_seq_seg[-1][0]:, i] for i in range(global_output_form[3])]).transpose()
    # Y_train_global = np.array([topK_next_interval(Y_train_global[:, i], interval_hours, 1) for i in range(global_output_form[3])]).transpose()
    #
    # Y_test = np.array([Y_test[local_train_seq_seg[-1][0]:, i] for i in range(local_output_form[3])]).transpose()
    # Y_test = np.array([topK_next_interval(Y_test[:, i], interval_hours, 1) for i in range(local_output_form[3])]).transpose()
    # Y_test_global = np.array([Y_test_global[global_train_seq_seg[-1][0]:, i] for i in range(global_output_form[3])]).transpose()
    # Y_test_global = np.array([topK_next_interval(Y_test_global[:, i], interval_hours, 1) for i in range(global_output_form[3])]).transpose()
    # ###############################################################################################################################################
    Y_real = np.copy(Y_test)
    Y_real = Y_real[:len(Y_test)]
    Y_real_global = np.copy(Y_test_global)
    Y_real_global = Y_real_global[:len(Y_test_global)]

    # Compute the size of an epoch.
    train_epoch_size = np.min([len(Y_train_global), len(X_train_global)])
    test_epoch_size = np.min([len(Y_test_global), len(X_test_global)])
    print('%d Training epoch size' % train_epoch_size)
    print('%d Testing epoch size' % test_epoch_size)

    # Make
    X_train_global = X_train_global[:train_epoch_size]
    X_train_local = X_train_local[:train_epoch_size]
    Y_train = Y_train[:train_epoch_size]
    Y_real = Y_real[:test_epoch_size]
    Y_train_global = Y_train_global[:train_epoch_size]
    Y_real_global = Y_real_global[:test_epoch_size]

    X_test_global = X_test_global[:test_epoch_size]
    X_test_local = X_test_local[:test_epoch_size]
    Y_test = Y_test[:test_epoch_size]
    Y_test_global = Y_test_global[:test_epoch_size]

    # Delete testing data with missing values since testing data cannot be imputed.
    # -----------------------------------------------------------------------------##################################################
    # local
    remain_list = []
    for i in range(len(Y_test)):
        # Check missing or not, if X_train_global[i] is missing, then this command will return True
        if not np.isnan(np.sum(Y_test[i])):
            remain_list.append(i)
    Y_test = Y_test[remain_list]
    Y_real = Y_real[remain_list]
    X_test_local = X_test_local[remain_list]
    Y_test_global = Y_test_global[remain_list]
    Y_real_global = Y_real_global[remain_list]
    X_test_global = X_test_global[remain_list]
    # global
    remain_list = []
    for i in range(len(Y_test_global)):
        # Check missing or not, if X_train_local[i] is missing, then this command will return True
        if not np.isnan(np.sum(Y_test_global[i])):
            remain_list.append(i)
    Y_test = Y_test[remain_list]
    Y_real = Y_real[remain_list]
    X_test_local = X_test_local[remain_list]
    Y_test_global = Y_test_global[remain_list]
    Y_real_global = Y_real_global[remain_list]
    X_test_global = X_test_global[remain_list]
    # -----------------------------------------------------------------------------

    # # local
    # i = 0
    # while i < len(Y_test):
    #     # Check missing or not, if Y_test[i] is missing, then this command will return True
    #     if np.isnan(sum(Y_test[i])):
    #         Y_test = np.delete(Y_test, i, 0)
    #         Y_real = np.delete(Y_real, i, 0)
    #         X_test = np.delete(X_test, i, 0)
    #         Y_test_global = np.delete(Y_test_global, i, 0)
    #         Y_real_global = np.delete(Y_real_global, i, 0)
    #         X_test_global = np.delete(X_test_global, i, 0)
    #         i = -1
    #     i += 1
    # # global
    # i = 0
    # while i < len(Y_test_global):
    #     # Check missing or not, if Y_test[i] is missing, then this command will return True
    #     if np.isnan(sum(Y_test_global[i])):
    #         Y_test = np.delete(Y_test, i, 0)
    #         Y_real = np.delete(Y_real, i, 0)
    #         X_test = np.delete(X_test, i, 0)
    #         Y_test_global = np.delete(Y_test_global, i, 0)
    #         Y_real_global = np.delete(Y_real_global, i, 0)
    #         X_test_global = np.delete(X_test_global, i, 0)
    #         i = -1
    #     i += 1
    # -----------------------------------------------------------------------------#################################################

    print('Delete invalid testing data, remain ', len(Y_test), 'test sequences')

    # Classification: Use interval to slice the variable Y in the training dataset.
    # Y_test_original = np.copy(Y_test)
    # local
    if local_output_form[2] == 1:
        for i in range(local_output_form[3]):
            Y_test[:, i] = np.array([(y - mean_y_train[i]) / std_y_train[i] for y in Y_test[:, i]])
    else:
        print('Classification: convert value to labels Y ..')
        new_Y_train = np.zeros(shape=(len(Y_train), local_output_form[2]*local_output_form[3]))
        interval_y = local_output_form[1] / local_output_form[2]
        for i in range(local_output_form[3]):
            for idx_row in range(len(Y_train)):
                # Given slice 0-2 2-4 4-6 6-8 8-10 10-12 12-14 14-16 ...
                # 13.6 => the 6-th entry is set to 1
                label_y = int(np.floor(Y_train[idx_row] / interval_y))
                new_Y_train[idx_row, i * local_output_form[2] + label_y] = 1

        Y_train = new_Y_train
        del new_Y_train

        new_Y_test = np.zeros(shape=(len(Y_test), local_output_form[2]*local_output_form[3]))
        interval_y = local_output_form[1] / local_output_form[2]
        for i in range(local_output_form[3]):
            for idx_row in range(len(Y_test)):
                # Given slice 0-2 2-4 4-6 6-8 8-10 10-12 12-14 14-16 ...
                # 13.6 => the 6-th entry is set to 1
                label_y = int(np.floor(Y_test[idx_row] / interval_y))
                new_Y_test[idx_row, i * local_output_form[2] + label_y] = 1

        Y_test = new_Y_test
        del new_Y_test,

    # global
    if global_output_form[2] == 1:
        for i in range(global_output_form[3]):
            Y_test_global[:, i] = np.array([(y - mean_y_train_global[i]) / std_y_train_global[i] for y in Y_test_global[:, i]])
    else:
        print('Classification: convert value to labels Y ..')
        new_Y_train_global = np.zeros(shape=(len(Y_train_global), global_output_form[2]*global_output_form[3]))
        interval_y = global_output_form[1] / global_output_form[2]
        for i in range(global_output_form[3]):
            for idx_row in range(len(Y_train)):
                # Given slice 0-2 2-4 4-6 6-8 8-10 10-12 12-14 14-16 ...
                # 13.6 => the 6-th entry is set to 1
                label_y_global = int(np.floor(Y_train_global[idx_row] / interval_y))
                new_Y_train_global[idx_row, i * global_output_form[2] + label_y_global] = 1

        Y_train_global = new_Y_train_global
        del new_Y_train_global

        new_Y_test_global = np.zeros(shape=(len(Y_test_global), global_output_form[2]*global_output_form[3]))
        interval_y = global_output_form[1] / global_output_form[2]
        for i in range(global_output_form[3]):
            for idx_row in range(len(Y_test)):
                # Given slice 0-2 2-4 4-6 6-8 8-10 10-12 12-14 14-16 ...
                # 13.6 => the 6-th entry is set to 1
                label_y_global = int(np.floor(Y_test_global[idx_row] / interval_y))
                new_Y_test_global[idx_row, i * global_output_form[2] + label_y_global] = 1

        Y_test_global = new_Y_test_global
        del new_Y_test_global


    #
    # ----- END: Data Pre-processing -----------------------------------------------------------------------------------
    #


    #
    # ----- START: Data Partition --------------------------------------------------------------------------------------
    #

    # Validation set
    num_split_to_valid = int(len(X_train_global)*valid_ratio)
    X_valid_global = X_train_global[-num_split_to_valid:]
    X_valid_local = X_train_local[-num_split_to_valid:]
    Y_valid = Y_train[-num_split_to_valid:]
    Y_valid_global = Y_train_global[-num_split_to_valid:]

    # Training set
    X_train_global = X_train_global[:-num_split_to_valid]
    X_train_local = X_train_local[:-num_split_to_valid]
    Y_train = Y_train[:-num_split_to_valid]
    Y_train_global = Y_train_global[:-num_split_to_valid]

    # Generate model_input2 from model_input
    #X_train2 = X_train[:, :, 2, 2, :].reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[4])
    #X_test2 = X_test[:, :, 2, 2, :].reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[4])

    # Generate model_input3 from model_input (LSTM)
    # X_train3 = X_train_global[:, :, 2, 2, :]
    # X_test3 = X_test_global[:, :, 2, 2, :]
    # [:, :, 2, 2, :]  -> [:, :, center_i, center_j, :]
    X_train_global3 = X_train_global[:, :, global_center_i, global_center_j, :]
    X_train_local3 = X_train_local[:, :, center_i, center_j, :]
    X_test_global3 = X_test_global[:, :, global_center_i, global_center_j, :]
    X_test_local3 = X_test_local[:, :, center_i, center_j, :]

    # Generate XGBoost model from model_input
    # ###################################################################################################################
    # local only
    # ###################################################################################################################
    X_train_XGB = np.zeros(shape=(X_train_local.shape[0], X_train_local.shape[1], len(target_site.adj_map) * X_train_local.shape[4]),
                           dtype='float32')
    X_test_XGB = np.zeros(shape=(X_test_local.shape[0], X_test_local.shape[1], len(target_site.adj_map) * X_test_local.shape[4]),
                          dtype='float32')
    indx = 0
    for i in target_site.adj_map:
        (x, y) = target_site.adj_map[i]
        X_train_XGB[:, :, indx*X_test_local.shape[4]:(indx+1)*X_test_local.shape[4]] = X_train_local[:, :, x, y, :]
        X_test_XGB[:, :, indx*X_test_local.shape[4]:(indx+1)*X_test_local.shape[4]] = X_test_local[:, :, x, y, :]
        indx += 1
    X_train_XGB = X_train_XGB.reshape(X_train_XGB.shape[0], X_train_XGB.shape[1]*X_train_XGB.shape[2])
    X_test_XGB = X_test_XGB.reshape(X_test_XGB.shape[0], X_test_XGB.shape[1]*X_test_XGB.shape[2])

    print('Take %d data to validation set' % num_split_to_valid)

    #
    # ----- END: Data Partition ----------------------------------------------------------------------------------------

    #  -----------------------------------------------------------------------------------------
    # Train Global Model
    # -----------------------------------------------------------------------------------------
    # g_model = GlobalForecastModel(target_site=target_site, output_form=global_output_form,
    #                               is_global_pretrained=False,
    #                               global_pollution_kind=global_pollution_kind,
    #                               global_target_kind=global_target_kind,
    #                               global_hyper_params=global_hyper_params,
    #                               global_train_seq_seg=global_train_seq_seg,
    #                               global_feature_kind_shift=6,
    #                               global_input_map_shape=global_site.shape)
    # global_model_path = ("global_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
    #                         % (target_site_name, training_year[0], training_start_point, training_year[-1],
    #                            training_end_point, interval_hours, global_target_kind))
    #
    # g_model.train(x_train_global=X_train_global, x_train_global2=X_train_global3,
    #               y_train_global=Y_train_global, x_test_global=X_test_global,
    #               x_test_global2=X_test_global3, y_test_global=Y_test_global,
    #               global_model_path=global_model_path)

    #  -----------------------------------------------------------------------------------------
    # Train Hybrid Model
    # -----------------------------------------------------------------------------------------
    # Setup label normalization parameters ##################################################################################
    norm_params = {'local': {'mean': mean_y_train.tolist(),
                             'std': std_y_train.tolist()},
                   'global': {'mean': mean_y_train_global.tolist(),
                              'std': std_y_train_global.tolist()}}

    f_model = HybridForecastModel(target_site=target_site, output_form=local_output_form,
                                  is_global_pretrained=False, is_local_pretrained=False,
                                  global_pollution_kind=global_pollution_kind, local_pollution_kind=local_pollution_kind,
                                  global_target_kind=global_target_kind, local_target_kind=local_target_kind,
                                  global_hyper_params=global_hyper_params, local_hyper_params=local_hyper_params,
                                  global_train_seq_seg=global_train_seq_seg, local_train_seq_seg=local_train_seq_seg,
                                  global_input_map_shape=global_site.shape, local_input_map_shape=target_site.shape)

    if not model_name:
        model_name = ("%s_training_%s_m%s_to_%s_m%s_interval_%s"
                      % (target_site_name, training_year[0], training_start_point, training_year[-1],
                         training_end_point, interval_hours))

    global_model_nn_path = ("global_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
                            % (target_site_name, training_year[0], training_start_point, training_year[-1],
                               training_end_point, interval_hours, global_target_kind))
    local_model_nn_path = ("local_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
                           % (target_site_name, training_year[0], training_start_point, training_year[-1],
                              training_end_point, interval_hours, local_target_kind))

    model_xgb_path = ("XGBoost_%s_training_%s_m%s_to_%s_m%s_interval_%s"
                      % (target_site_name, training_year[0], training_start_point, training_year[-1], training_end_point,
                         interval_hours))

    model_ensemble_path = ("ensemble_%s_training_%s_m%s_to_%s_m%s_interval_%s"
                           % (target_site_name, training_year[0], training_start_point, training_year[-1],
                              training_end_point, interval_hours))

    check_folder2(os.path.join(model_root_path, model_name))

    # Save parameters
    save_params = {'interval_hours': interval_hours, 'norm_params': norm_params,
                   'global_hyper_params': global_hyper_params, 'local_hyper_params': local_hyper_params,
                   'global_train_seq_seg': global_train_seq_seg, 'local_train_seq_seg': local_train_seq_seg}
    with open(os.path.join(model_root_path, model_name, 'model_conf.json'), 'w') as f:
        json.dump(save_params, f)

    # Set all model path
    global_model_nn_path = os.path.join(model_root_path, model_name, global_model_nn_path)
    local_model_nn_path = os.path.join(model_root_path, model_name, local_model_nn_path)
    model_xgb_path = os.path.join(model_root_path, model_name, model_xgb_path)
    model_ensemble_path = os.path.join(model_root_path, model_name, model_ensemble_path)

    # parameters:
    # x_train_xgb,
    # x_train_global, x_train_global2, x_train_local, x_train_local2,
    # y_train_global, y_train_local,
    # x_test_global, x_test_global2, x_test_local, x_test_local2,
    # y_test_global, y_test_local,
    # global_model_path, local_model_path, model_xgb_path, model_ensemble_path

    f_model.train(X_train_XGB,
                  X_train_global, X_train_global3, X_train_local, X_train_local3,
                  Y_train_global, Y_train,
                  X_test_global, X_test_global3, X_test_local, X_test_local3,
                  Y_test_global, Y_test,
                  global_model_nn_path, local_model_nn_path, model_xgb_path, model_ensemble_path,
                  # Finishing function call here to pass over pre-train EPA model
                  x_full=X_train_global_full, x3_full=X_train_global3_full,
                  x_valid_full=X_valid_global_full, x_valid3_full=X_valid_global3_full,
                  y_full=Y_train_global_full, y_valid_full=Y_valid_global_full)

    # Set label normalization parameters
    f_model.set_norm(norm_params)
    # Inference
    pred, global_pred = f_model.evaluate(x_test_xgb=X_test_XGB, x_test_global=X_test_global,
                                         x_test_global2=X_test_global3,
                                         x_test_local=X_test_local, x_test_local2=X_test_local3)

    print()

    mse = np.mean((Y_real - pred) ** 2)
    print('local testing MSE:{0}'.format(mse))
    # Save evaluate result csv file
    with open('/media/clliao/006a3168-df49-4b0a-a874-891877a888701/TCH/pm25_rlt/local_rlt.csv', 'w') as f:
        f.write('real,pred\n')
        for i in range(len(Y_real)):
            f.write('{0},{1}\n'.format(Y_real[i][0], pred[i][0]))

    mse = np.mean((Y_real_global - global_pred) ** 2)
    print('global testing MSE:{0}'.format(mse))

    with open('/media/clliao/006a3168-df49-4b0a-a874-891877a888701/TCH/pm25_rlt/global_rlt.csv', 'w') as f:
        f.write('real,pred\n')
        for i in range(len(Y_real_global)):
            f.write('{0},{1}\n'.format(Y_real_global[i][0], global_pred[i][0]))
