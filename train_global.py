import time

from ConvLSTM.global_forecast_model import GlobalForecastModel
from ConvLSTM.config import *

from utility.reader import read_global_data_map, construct_time_map2
from utility.missing_value_processer import missing_check
from utility.Utilities import *
from ConvLSTM.config import root, dataset_path, site_map2

#
# ----- START: Parameters Declaration ----------------------------------------------------------------------------------
#

# Define data loading parameters
data_update = False
# Notice that 'WIND_SPEED' must be the last feature and 'WIND_DIREC' must be the last second.
pollution_kind = ['PM2.5', 'O3', 'SO2', 'CO', 'NOx', 'NO', 'NO2', 'AMB_TEMP', 'RH',
                  'PM2.5_x_O3', 'PM2.5_x_CO', 'PM2.5_x_NOx', 'O3_x_CO', 'O3_x_NOx', 'O3_x_AMB_TEMP', 'CO_x_NOx',
                  'WIND_SPEED', 'WIND_DIREC']
# target_kind = 'PM2.5'
target_kind = ['PM2.5', 'O3', 'NO', 'NO2', 'NOx']

# Define target duration for training
training_year = [2014, 2016]  # change format from   2014-2015   to   ['2014', '2015']
training_duration = ['1/1', '12/31']

# Define target duration for prediction
testing_year = [2016, 2017]
testing_duration = ['12/29', '1/31']
interval_hours = 12  # predict the average of the subsequent hours as the label, set to 1 as default.
is_training = True  # True False

# Define model parameters
hyper_params = {
        'num_filters': 16,
        'kernel_size': (3, 3),
        'regularizer': 1e-7,
        'cnn_dropout': 0.5,
        'r_dropout': 0.5,
        'pool_size': (2, 2),
        'epoch': 50,
        'batch_size': 256,
        'interval_hours': interval_hours
    }

# Define the output form
output_form = (0, 80, 1, len(target_kind))      # Regression (min, max, num_of_slice, num_of_obj)
# output_form = (0, 150, 50, 1)                 # Classification (min, max, num_of_slice, num_of_obj)
if output_form[2] > 1:
    mean_form = [i for i in range(output_form[0], output_form[1], int(output_form[1]/output_form[2]))]
    mean_form = [i+int(output_form[1]/output_form[2])/2 for i in mean_form]
output_size = output_form[2] * output_form[3]

# Define target site and its adjacent map for prediction
pollution_site_map2 = site_map2()

for target_site_keys in pollution_site_map2:
    print(pollution_site_map2[target_site_keys].site_name)

    target_site = pollution_site_map2[target_site_keys]
    center_i = int(target_site.shape[0]/2)
    center_j = int(target_site.shape[1]/2)
    local = target_site.local
    city = target_site.city
    target_site_name = target_site.site_name
    site_list = list(target_site.adj_map.keys())  # ['士林', '中山', '松山', '汐止', '板橋', '萬華', '古亭', '土城', '新店']
    map_shape = target_site.shape

    # Define pre-processing parameters
    # 'day of year', 'day of week' and 'time of day' respectively are represented by two dimensions
    feature_kind_shift = 6

    train_seq_seg = [(36, 1)]   # [(12, 1), (24, 2), (48, 3), (72, 6)]
    train_seq_length = int(train_seq_seg[0][0] / train_seq_seg[0][1])
    for seg_idx in range(1, len(train_seq_seg)):
        train_seq_length += int((train_seq_seg[seg_idx][0] - train_seq_seg[seg_idx-1][0]) / train_seq_seg[seg_idx][1])
    #
    # ----- END: Parameters Declaration --------------------------------------------------------------------------------
    #


    #
    # ----- START: Year Processing -------------------------------------------------------------------------------------
    #

    # Clear redundant year, i.e., [2014, 2014] ==> [2014]
    if training_year[0] == training_year[1]:
        training_year.pop(1)
    if testing_year[0] == testing_year[1]:
        testing_year.pop(1)
    else:
        print('The range of testing year should not be more than one '
              'year or cross contiguous years.')

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
    print('Target: %s' % target_kind)

    # Set start time of data loading.
    print('Loading data .. ')
    start_time = time.time()
    initial_time = time.time()

    # Load training data, where: size(X_train) = (data_size, map_l, map_w, map_h), not sequentialized yet.
    print('Preparing training dataset ..')
    X_train = read_global_data_map(path=data_path, site=target_site, feature_selection=pollution_kind,
                                   date_range=np.atleast_1d(training_year), beginning=training_duration[0],
                                   finish=training_duration[-1], update=data_update)
    X_train = missing_check(X_train)
    Y_train = np.array(X_train)[:, center_i, center_j, [6 + pollution_kind.index(i) for i in target_kind]]

    # Load testing data, where: size(X_test) = (data_size, map_l, map_w, map_h), not sequentialized yet.
    print('Preparing testing dataset ..')
    X_test = read_global_data_map(path=data_path, site=target_site, feature_selection=pollution_kind,
                                  date_range=np.atleast_1d(testing_year), beginning=testing_duration[0],
                                  finish=testing_duration[-1], update=data_update)
    X_test = missing_check(X_test)
    Y_test = np.array(X_test)[:, center_i, center_j, [6 + pollution_kind.index(i) for i in target_kind]]

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
    X_train = construct_time_map2(X_train[:-1], train_seq_seg)
    X_test = construct_time_map2(X_test[:-1], train_seq_seg)
    final_time = time.time()
    time_spent_printer(start_time, final_time)

    # Regression: Normalize the dependent variable Y in the training dataset.
    mean_y_train = np.zeros(shape=(output_form[3],))
    std_y_train = np.zeros(shape=(output_form[3],))
    if output_form[2] == 1:
        for i in range(output_form[3]):
            print('Regression: Normalize Y_train[%d] ..' % i)

            mean_y_train[i] = np.mean(Y_train[:, i])
            std_y_train[i] = np.std(Y_train[:, i])
            if not std_y_train[i]:
                input("Denominator cannot be 0.")
            Y_train[:, i] = np.array([(y - mean_y_train[i]) / std_y_train[i] for y in Y_train[:, i]])
            print('mean_y_train: %f  std_y_train: %f' % (mean_y_train[i], std_y_train[i]))
            print('Feature processing ..')

    # Construct corresponding label.
    Y_train = np.array([Y_train[train_seq_seg[-1][0]:, i] for i in range(output_form[3])]).transpose()
    Y_train = np.array([topK_next_interval(Y_train[:, i], interval_hours, 1) for i in range(output_form[3])]).transpose()
    Y_test = np.array([Y_test[train_seq_seg[-1][0]:, i] for i in range(output_form[3])]).transpose()
    Y_test = np.array([topK_next_interval(Y_test[:, i], interval_hours, 1) for i in range(output_form[3])]).transpose()
    Y_real = np.copy(Y_test)
    Y_real = Y_real[:len(Y_test)]

    # Compute the size of an epoch.
    train_epoch_size = np.min([len(Y_train), len(X_train)])
    test_epoch_size = np.min([len(Y_test), len(X_test)])
    print('%d Training epoch size' % train_epoch_size)
    print('%d Testing epoch size' % test_epoch_size)

    # Make
    X_train = X_train[:train_epoch_size]
    Y_train = Y_train[:train_epoch_size]
    Y_real = Y_real[:test_epoch_size]
    X_test = X_test[:test_epoch_size]
    Y_test = Y_test[:test_epoch_size]

    # Delete testing data with missing values since testing data cannot be imputed.
    i = 0
    while i < len(Y_test):
        # Check missing or not, if Y_test[i] is missing, then this command will return True
        if not(Y_test[i, 1] > -10000):
            Y_test = np.delete(Y_test, i, 0)
            Y_real = np.delete(Y_real, i, 0)
            X_test = np.delete(X_test, i, 0)
            i = -1
        i += 1

    print('Delete invalid testing data, remain ', len(Y_test), 'test sequences')

    # Classification: Use interval to slice the variable Y in the training dataset.
    Y_test_original = np.copy(Y_test)
    if output_form[2] == 1:
        for i in range(output_form[3]):
            Y_test[:, i] = np.array([(y - mean_y_train[i]) / std_y_train[i] for y in Y_test[:, i]])
    else:
        print('Classification: convert value to labels Y ..')
        new_Y_train = np.zeros(shape=(len(Y_train), output_form[2]*output_form[3]))
        interval_y = output_form[1] / output_form[2]
        for i in range(output_form[3]):
            for idx_row in range(len(Y_train)):
                # Given slice 0-2 2-4 4-6 6-8 8-10 10-12 12-14 14-16 ...
                # 13.6 => the 6-th entry is set to 1
                label_y = int(np.floor(Y_train[idx_row] / interval_y))
                new_Y_train[idx_row, i * output_form[2] + label_y] = 1
        Y_train = new_Y_train
        del new_Y_train

        new_Y_test = np.zeros(shape=(len(Y_test), output_form[2]*output_form[3]))
        interval_y = output_form[1] / output_form[2]
        for i in range(output_form[3]):
            for idx_row in range(len(Y_test)):
                # Given slice 0-2 2-4 4-6 6-8 8-10 10-12 12-14 14-16 ...
                # 13.6 => the 6-th entry is set to 1
                label_y = int(np.floor(Y_test[idx_row] / interval_y))
                new_Y_test[idx_row, i * output_form[2] + label_y] = 1
        Y_test = new_Y_test
        del new_Y_test

    #
    # ----- END: Data Pre-processing -----------------------------------------------------------------------------------
    #


    #
    # ----- START: Data Partition --------------------------------------------------------------------------------------
    #

    # Validation set
    X_valid = X_train[-800:]
    Y_valid = Y_train[-800:]

    # Training set
    X_train = X_train[:-800]
    Y_train = Y_train[:-800]

    # Generate model_input2 from model_input
    #X_train2 = X_train[:, :, 2, 2, :].reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[4])
    #X_test2 = X_test[:, :, 2, 2, :].reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[4])

    # Generate model_input3 from model_input
    X_train3 = X_train[:, :, 2, 2, :]
    X_test3 = X_test[:, :, 2, 2, :]

    # Generate XGBoost model from model_input
    X_train_XGB = np.zeros(shape=(X_train.shape[0], X_train.shape[1], len(target_site.adj_map) * X_train.shape[4]),
                           dtype='float32')
    X_test_XGB = np.zeros(shape=(X_test.shape[0], X_test.shape[1], len(target_site.adj_map) * X_test.shape[4]),
                          dtype='float32')
    indx = 0
    for i in target_site.adj_map:
        (x, y) = target_site.adj_map[i]
        X_train_XGB[:, :, indx*X_test.shape[4]:(indx+1)*X_test.shape[4]] = X_train[:, :, x, y, :]
        X_test_XGB[:, :, indx*X_test.shape[4]:(indx+1)*X_test.shape[4]] = X_test[:, :, x, y, :]
        indx += 1
    X_train_XGB = X_train_XGB.reshape(X_train_XGB.shape[0], X_train_XGB.shape[1]*X_train_XGB.shape[2])
    X_test_XGB = X_test_XGB.reshape(X_test_XGB.shape[0], X_test_XGB.shape[1]*X_test_XGB.shape[2])

    print('Take 800 data to validation set')

    #
    # ----- END: Data Partition ----------------------------------------------------------------------------------------
    #

    f_model = GlobalForecastModel(pollution_kind, target_kind, target_site, feature_kind_shift,
                                  train_seq_seg, hyper_params, (X_train.shape[2], X_train.shape[3]), output_form)

    model_nn_path = ("Ensemble_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
                     % (target_site_name, training_year[0], training_start_point, training_year[-1],
                        training_end_point, interval_hours, target_kind))

    model_xgb_path = ("XGBoost_%s_training_%s_m%s_to_%s_m%s_interval_%s"
                      % (target_site, training_year[0], training_start_point, training_year[-1], training_end_point,
                         interval_hours))

    model_ensemble_path = ("ensemble_%s_training_%s_m%s_to_%s_m%s_interval_%s"
                           % (target_site, training_year[0], training_start_point, training_year[-1],
                              training_end_point, interval_hours))

    f_model.train(X_train_XGB, X_train, X_train3, Y_train,
                  X_test, X_test3, Y_test,
                  model_nn_path, model_xgb_path, model_ensemble_path)
