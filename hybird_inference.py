import time
import json
from datetime import datetime, timedelta

from ConvLSTM.hybrid_forecast_model import HybridForecastModel
from ConvLSTM.config import *

from utility.reader import read_hybrid_data_map, construct_time_map2, read_global_or_local_data_map, \
    construct_time_map_with_label, construct_time_map_without_label
from utility.missing_value_processer import missing_check, drop_missing
from utility.Utilities import *
from ConvLSTM.config import root, model_root_path, site_map2, pollution_site_local_map, global_site_lock


def hybrid_inference(year, date, hour, local, city, site_name, interval_hours, model_path):
    name_interval = '%dh' % interval_hours
    full_path = os.path.join(model_path, local, city, site_name, name_interval)
    # param_path = "/media/clliao/006a3168-df49-4b0a-a874-891877a888701/AirQuality/PM25Forecast/ConvLSTM/models/Node_09_training_2018_m9_to_2018_m9_interval_1/model_conf.json"
    param_path = os.path.join(full_path, "model_conf.json")

    with open(param_path, 'r') as f:
        params = json.load(f)

    global_feature_kind_shift = 6
    local_feature_kind_shift = 6

    global_pollution_kind = params['global_pollution_kind']
    local_pollution_kind = params['local_pollution_kind']

    pollution_kind = [global_pollution_kind, local_pollution_kind]

    local_target_kind = ['PM2_5']
    global_target_kind = local_target_kind

    global_hyper_params = params['global_hyper_params']
    local_hyper_params = params['local_hyper_params']

    norm_params = params['norm_params']

    local_output_form = (0, 80, 1, len(local_target_kind))  # Regression (min, max, num_of_slice, num_of_obj)
    global_output_form = (0, 80, 1, len(global_target_kind))  # Regression (min, max, num_of_slice, num_of_obj)

    pollution_site_map2 = site_map2()

    # global
    global_site = pollution_site_map2[global_site_lock]  # 龍潭
    global_center_i = int(global_site.shape[0] / 2)
    global_center_j = int(global_site.shape[1] / 2)

    # local
    target_site = pollution_site_local_map[site_name]
    center_i = int(target_site.shape[0] / 2)
    center_j = int(target_site.shape[1] / 2)
    local = target_site.local
    city = target_site.city
    target_site_name = target_site.site_name
    site_list = sorted(list(target_site.adj_map.keys()))  # ['士林', '中山', '松山', '汐止', '板橋', '萬華', '古亭', '土城', '新店']
    map_shape = target_site.shape

    global_train_seq_seg = params['global_train_seq_seg']  # [(12, 1), (24, 2), (48, 3), (72, 6)]
    local_train_seq_seg = params['local_train_seq_seg']

    train_seq_length = int(global_train_seq_seg[0][0] / global_train_seq_seg[0][1])
    for seg_idx in range(1, len(global_train_seq_seg)):
        train_seq_length += int(
            (global_train_seq_seg[seg_idx][0] - global_train_seq_seg[seg_idx - 1][0]) / global_train_seq_seg[seg_idx][
                1])

    #
    # ----- END: Parameters Declaration --------------------------------------------------------------------------------
    #

    #
    # ----- START: datetime Processing ---------------------------------------------------------------------------------
    #

    testing_year = [year]

    # Generate years sequence, i.e., [2014, 2016] ==> [2014, 2015, 2016]
    range_of_year = testing_year[-1] - testing_year[0]
    for i in range(range_of_year):
        if not(int(i + testing_year[0]) in testing_year):
            testing_year.insert(i, int(i + testing_year[0]))

    ###########################################
    target_time = datetime.strptime(str(year) + '/' + date, '%Y/%m/%d')
    target_time_delta = target_time - timedelta(hours=train_seq_length)
    testing_duration = [datetime.strftime(target_time_delta, '%m/%d'),
                        datetime.strftime(target_time, '%m/%d')]
    ###########################################

    #
    # ----- END: Year Processing ---------------------------------------------------------------------------------------
    #


    #
    # ----- START: Data Loading ----------------------------------------------------------------------------------------
    #

    print('Testing for %s/%s to %s/%s' % (testing_year[0], testing_duration[0], testing_year[-1], testing_duration[-1]))

    print('Loading data .. ')
    start_time = time.time()
    initial_time = time.time()

    # Load testing data, where: size(X_test) = (data_size, map_l, map_w, map_h), not sequentialized yet.
    print('Preparing testing dataset ..')
    X_test_global, X_test_local = read_hybrid_data_map(site=target_site,
                                                       feature_selection=pollution_kind,
                                                       date_range=np.atleast_1d(testing_year),
                                                       beginning=testing_duration[0],
                                                       finish=testing_duration[-1],
                                                       hour=hour,
                                                       seq_length=train_seq_length)
    X_test_global = missing_check(X_test_global)
    X_test_local = missing_check(X_test_local)
    # Y_test_global = np.array(X_test_global)[:, global_center_i, global_center_j, [local_feature_kind_shift + global_pollution_kind.index(i) for i in global_target_kind]]
    # Y_test = np.array(X_test_local)[:, center_i, center_j, [local_feature_kind_shift + local_pollution_kind.index(i) for i in local_target_kind]] # local

    #
    # ----- END: Data Loading ------------------------------------------------------------------------------------------
    #


    #
    # ----- START: Data Pre-processing ---------------------------------------------------------------------------------
    #

    # Construct sequential data.
    print('Construct time series dataset ..')
    start_time = time.time()

    X_test_global = construct_time_map_without_label(X_test_global[:-1], global_train_seq_seg, time_unit=60)
    X_test_local = construct_time_map_without_label(X_test_local[:-1], local_train_seq_seg, time_unit=60)
    final_time = time.time()
    time_spent_printer(start_time, final_time)


    # Delete testing data with missing values since testing data cannot be imputed.
    # -----------------------------------------------------------------------------
    # remain_list = drop_missing(X_test_local)
    # X_test_local = X_test_local[remain_list]
    # X_test_global = X_test_global[remain_list]
    # # global
    # remain_list = drop_missing(X_test_global)
    # X_test_local = X_test_local[remain_list]
    # X_test_global = X_test_global[remain_list]

    if np.isnan(np.mean(X_test_local)) or np.isnan(np.mean(X_test_global)):
        print('There is nan value in data!!!')
        return np.nan

    # -----------------------------------------------------------------------------

    #
    # ----- END: Data Pre-processing -----------------------------------------------------------------------------------
    #


    #
    # ----- START: Data Partition --------------------------------------------------------------------------------------
    #
    # Generate model_input3 from model_input (LSTM)
    X_test_global3 = X_test_global[:, :, global_center_i, global_center_j, :]
    X_test_local3 = X_test_local[:, :, center_i, center_j, :]

    # Generate XGBoost model from model_input
    # ###################################################################################################################
    # local only
    # ###################################################################################################################
    X_test_XGB = np.zeros(shape=(X_test_local.shape[0], X_test_local.shape[1], len(target_site.adj_map) * X_test_local.shape[4]),
                          dtype='float32')
    indx = 0
    for i in site_list:
        (x, y) = target_site.adj_map[i]
        X_test_XGB[:, :, indx*X_test_local.shape[4]:(indx+1)*X_test_local.shape[4]] = X_test_local[:, :, x, y, :]
        indx += 1
    X_test_XGB = X_test_XGB.reshape(X_test_XGB.shape[0], X_test_XGB.shape[1]*X_test_XGB.shape[2])
    #
    # ----- END: Data Partition ----------------------------------------------------------------------------------------

    f_model = HybridForecastModel(target_site=target_site, output_form=local_output_form,
                                  is_global_pretrained=False, is_local_pretrained=False,
                                  global_pollution_kind=global_pollution_kind, local_pollution_kind=local_pollution_kind,
                                  global_target_kind=global_target_kind, local_target_kind=local_target_kind,
                                  global_hyper_params=global_hyper_params, local_hyper_params=local_hyper_params,
                                  global_train_seq_seg=global_train_seq_seg, local_train_seq_seg=local_train_seq_seg,
                                  global_input_map_shape=global_site.shape, local_input_map_shape=target_site.shape)


    # global_model_nn_path = "/media/clliao/006a3168-df49-4b0a-a874-891877a888701/AirQuality/PM25Forecast/ConvLSTM/models/Node_09_training_2018_m9_to_2018_m9_interval_1/global_Node_09_training_2018_m9_to_2018_m9_interval_1_['PM2_5']"
    # local_model_nn_path = "/media/clliao/006a3168-df49-4b0a-a874-891877a888701/AirQuality/PM25Forecast/ConvLSTM/models/Node_09_training_2018_m9_to_2018_m9_interval_1/local_Node_09_training_2018_m9_to_2018_m9_interval_1_['PM2_5']"
    # model_xgb_path = "/media/clliao/006a3168-df49-4b0a-a874-891877a888701/AirQuality/PM25Forecast/ConvLSTM/models/Node_09_training_2018_m9_to_2018_m9_interval_1/XGBoost_Node_09_training_2018_m9_to_2018_m9_interval_1"
    # model_ensemble_path = "/media/clliao/006a3168-df49-4b0a-a874-891877a888701/AirQuality/PM25Forecast/ConvLSTM/models/Node_09_training_2018_m9_to_2018_m9_interval_1/ensemble_Node_09_training_2018_m9_to_2018_m9_interval_1"

    model_list = os.listdir(full_path)
    for model_name in model_list:
        if 'global' in model_name:
            global_model_nn_path = os.path.join(full_path, model_name)
        elif 'local' in model_name:
            local_model_nn_path = os.path.join(full_path, model_name)
        elif 'XGB' in model_name:
            model_xgb_path = os.path.join(full_path, model_name)
        elif 'ensemble' in model_name:
            model_ensemble_path = os.path.join(full_path, model_name)

    f_model.load_model(global_model_path=global_model_nn_path,
                       local_model_path=local_model_nn_path,
                       xgb_model_path=model_xgb_path,
                       ens_model_path=model_ensemble_path)

    # Set label normalization parameters
    f_model.set_norm(norm_params)

    # Inference
    pred, global_pred = f_model.evaluate(x_test_xgb=X_test_XGB, x_test_global=X_test_global,
                                         x_test_global2=X_test_global3,
                                         x_test_local=X_test_local, x_test_local2=X_test_local3)

    return pred


if __name__ == '__main__':
    predict = hybrid_inference(year=2018, date='9/5', hour='8', local='桃園', city='中科院', site_name='Node_09',
                               interval_hours=1, model_path=model_root_path)
    print(predict)

