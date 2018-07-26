# tornado server

import json
import tornado.ioloop
import tornado.options
import tornado.web
import os
import time

from ConvLSTM.hybrid_forecast_model import HybridForecastModel
from ConvLSTM.config import *

import logging
from logging.config import fileConfig


class MainHandler(tornado.web.RequestHandler):

    # sess, dict_current_tracks, model_track_recognition, model_track_prediction
    def initialize(self, args_dict):
        self.dict_data = args_dict['dict_data']
        self.dict_forecasters = args_dict['dict_forecasters']


    # Process POST Request and output text message
    def post(self):
        data = json.loads(self.request.body.decode('utf-8'))
        print('Got JSON data:', data)

        site_name = data[0]
        forecast = []
        for interval_hours in self.interval_hours_set:
            forecast.append(self.dict_forecasters[(site_name, interval_hours)])

        str_json = "{"
        for track_id in self.dict_current_tracks.keys():
            track_feature = self.dict_current_tracks[track_id]
            track_type, track_prob = self.model_track_recognition(track_feature)
            track_pos = self.model_track_prediction(track_feature)

            str_json += "\'track_id\':[" + str(track_type) + ", " + str(track_prob) + ", " + str(track_pos) + "], "

        str_json = "}"
        self.write(str_json)


# Define the dispatching mechanism of Request
def make_app(dict_data, dict_forecasters):
    args_dict = {
        "dict_data": dict_data,
        "dict_forecasters": dict_forecasters,
    }
    application = tornado.web.Application([(r"/", MainHandler, dict(args_dict=args_dict))])
    http_server = tornado.httpserver.HTTPServer(application)
    return http_server


def main(configure):

    tornado.options.parse_command_line()

    # Log
    fileConfig(configure.log_config)

    hyper_params = dict()
    dict_data = dict()
    dict_forecasters = dict()

    #
    # ----------- Global data setting (data from EPA) -----------
    #
    # Notice that 'WIND_SPEED' must be the last feature and 'WIND_DIREC' must be the last second.
    global_pollution_kind = ['PM2.5', 'O3', 'SO2', 'CO', 'NOx', 'NO', 'NO2', 'AMB_TEMP', 'RH',
                             'PM2.5_x_O3', 'PM2.5_x_CO', 'PM2.5_x_NOx', 'O3_x_CO', 'O3_x_NOx', 'O3_x_AMB_TEMP',
                             'CO_x_NOx', 'WIND_SPEED', 'WIND_DIREC']
    global_target_kind = ['PM2.5', 'O3', 'NO', 'NO2', 'NOx']

    global_hyper_params = {
        'num_filters': 16,
        'kernel_size': (3, 3),
        'regularizer': 1e-7,
        'cnn_dropout': 0.5,
        'r_dropout': 0.5,
        'pool_size': (2, 2),
        'epoch': 50,
        'batch_size': 256,
        'interval_hours': 1
    }

    global_train_seq_seg = [(36, 1)]  # [(12, 1), (24, 2), (48, 3), (72, 6)]
    global_train_seq_length = int(global_train_seq_seg[0][0] / global_train_seq_seg[0][1])
    for seg_idx in range(1, len(global_train_seq_seg)):
        global_train_seq_length += int((global_train_seq_seg[seg_idx][0] -
                                        global_train_seq_seg[seg_idx - 1][0]) / global_train_seq_seg[seg_idx][1])


    #
    # ----------- Local data setting (data from ) -----------
    #
    # Notice that 'WIND_SPEED' must be the last feature and 'WIND_DIREC' must be the last second.
    local_pollution_kind = ['PM2.5', 'O3', 'SO2', 'CO', 'NOx', 'NO', 'NO2', 'AMB_TEMP', 'RH',
                            'PM2.5_x_O3', 'PM2.5_x_CO', 'PM2.5_x_NOx', 'O3_x_CO', 'O3_x_NOx', 'O3_x_AMB_TEMP',
                            'CO_x_NOx', 'WIND_SPEED', 'WIND_DIREC']
    local_target_kind = ['PM2.5']

    local_hyper_params = {
        'num_filters': 16,
        'kernel_size': (3, 3),
        'regularizer': 1e-7,
        'cnn_dropout': 0.5,
        'r_dropout': 0.5,
        'pool_size': (2, 2),
        'epoch': 50,
        'batch_size': 256,
        'interval_hours': 1
    }

    local_train_seq_seg = [(36, 1)]  # [(12, 1), (24, 2), (48, 3), (72, 6)]
    local_train_seq_length = int(local_train_seq_seg[0][0] / local_train_seq_seg[0][1])
    for seg_idx in range(1, len(local_train_seq_seg)):
        local_train_seq_length += int((local_train_seq_seg[seg_idx][0] -
                                       local_train_seq_seg[seg_idx - 1][0]) / local_train_seq_seg[seg_idx][1])

    # Define the output form
    final_target_kind = ['PM2.5']
    output_form = (0, 80, 1, len(final_target_kind))  # Regression (min, max, num_of_slice, num_of_obj)

    feature_kind_shift = 6
    interval_hours_set = [1, 3, 6, 12]
    pollution_site_map2 = site_map2()
    for interval_hours in interval_hours_set:
        for target_site_keys in pollution_site_map2:
            target_site = pollution_site_map2[target_site_keys]
            print(target_site.site_name)

            hyper_params['interval_hours'] = interval_hours
            f_model = HybridForecastModel(pollution_kind, target_kind, target_site, feature_kind_shift,
                                          train_seq_seg, hyper_params, output_form)

            f_model = HybridForecastModel(global_pollution_kind, global_target_kind, global_feature_kind_shift,
                                          global_train_seq_seg, global_hyper_params, global_input_map_shape,
                                          local_pollution_kind, local_target_kind, local_feature_kind_shift,
                                          local_train_seq_seg, local_hyper_params, local_input_map_shape,
                                          output_form)

            dict_forecasters[(target_site.site_name, interval_hours)] = f_model

    # service
    serv = make_app(dict_data, dict_forecasters)

    # Set port to 8080
    serv.listen(8080)

    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    main()

