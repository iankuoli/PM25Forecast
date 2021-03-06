import keras
from keras.layers import Input, Conv2D, LSTM, MaxPooling2D, MaxPooling3D, concatenate, Bidirectional, Activation, TimeDistributed, Dense, Dropout, Flatten, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor

import pickle
import time
import numpy as np

from ConvLSTM.config import root, dataset_path, site_map2
from utility.Utilities import *


class HybridForecastModel:

    #
    # Initialize forecast model and declare LSTM & ConvLSTM model structures
    #
    def __init__(self, target_site, output_form,
                 is_global_pretrained, is_local_pretrained,
                 global_pollution_kind, local_pollution_kind,
                 global_target_kind, local_target_kind,
                 global_hyper_params, local_hyper_params,
                 global_train_seq_seg=(36, 1), local_train_seq_seg=(360, 30),
                 global_feature_kind_shift=6, local_feature_kind_shift=6,
                 global_input_map_shape=(5, 5), local_input_map_shape=(12, 12)):

        # Target site
        self.target_site = target_site
        self.is_global_pretrained = is_global_pretrained
        self.is_local_pretrained = is_local_pretrained

        # Global hyper-parameters
        self.global_pollution_kind = global_pollution_kind
        self.global_target_kind = global_target_kind
        self.global_feature_kind_shift = global_feature_kind_shift
        self.global_train_seq_seg = global_train_seq_seg
        self.global_train_seq_length = int(self.global_train_seq_seg[0][0] / self.global_train_seq_seg[0][1])
        self.global_mean_y_train = np.zeros(len(self.global_target_kind))
        self.global_std_y_train = np.zeros(len(self.global_target_kind))


        self.global_num_filters = global_hyper_params['num_filters']
        self.global_kernel_size = global_hyper_params['kernel_size']
        self.global_regularizer = global_hyper_params['regularizer']
        self.global_cnn_dropout = global_hyper_params['cnn_dropout']
        self.global_r_dropout = global_hyper_params['r_dropout']
        self.global_pool_size = global_hyper_params['pool_size']
        self.global_3d_pool_size = global_hyper_params['3d_pool_size']
        self.global_epoch = global_hyper_params['epoch']
        self.global_batch_size = global_hyper_params['batch_size']
        self.global_interval_hours = global_hyper_params['interval_hours']
        self.global_input_map_shape = global_input_map_shape

        # Local hyper-parameters
        self.local_pollution_kind = local_pollution_kind
        self.local_target_kind = local_target_kind
        self.local_feature_kind_shift = local_feature_kind_shift
        self.local_train_seq_seg = local_train_seq_seg
        self.local_train_seq_length = int(self.local_train_seq_seg[0][0] / self.local_train_seq_seg[0][1])
        self.local_mean_y_train = np.zeros(len(self.local_target_kind))
        self.local_std_y_train = np.zeros(len(self.local_target_kind))

        self.local_num_filters = local_hyper_params['num_filters']
        self.local_kernel_size = local_hyper_params['kernel_size']
        self.local_regularizer = local_hyper_params['regularizer']
        self.local_cnn_dropout = local_hyper_params['cnn_dropout']
        self.local_r_dropout = local_hyper_params['r_dropout']
        self.local_pool_size = local_hyper_params['pool_size']
        self.local_epoch = local_hyper_params['epoch']
        self.local_batch_size = local_hyper_params['batch_size']
        self.local_interval_hours = local_hyper_params['interval_hours']
        self.local_input_map_shape = local_input_map_shape

        # Forecast result
        self.forecast = np.zeros(self.global_interval_hours)

        # Define the output form
        # output_form = (0, 80, 1, len(self.target_kind))     # Regression (min, max, num_of_slice, num_of_obj)
        # output_form = (0, 150, 50, 1)                       # Classification (min, max, num_of_slice, num_of_obj)
        if output_form[2] > 1:
            mean_form = [i for i in range(output_form[0], output_form[1], int(output_form[1] / output_form[2]))]
            mean_form = [i + int(output_form[1] / output_form[2]) / 2 for i in mean_form]
        output_size = output_form[2] * output_form[3]

        # Specify the path where model will be saved.
        print('Build the forecast model...')

        self.xgb_model = None
        self.ensemble_model = None

        #
        # ----- START: Global Models (LSTM & ConvLSTM) Definition ------------------------------------------------------
        #
        # feature 'WIND_DIREC' has two dimension
        global_input_size = len(self.global_pollution_kind) + 1 if 'WIND_DIREC' in self.global_pollution_kind else len(self.global_pollution_kind)
        global_input_size = global_input_size + self.global_feature_kind_shift
        global_input_shape = (self.global_train_seq_length,
                              self.global_input_map_shape[0], self.global_input_map_shape[1], global_input_size)

        # Input for ConvLSTM: 5D tensor with shape: (samples_index, sequence_index, row, col, feature/channel)
        global_model_input = Input(shape=global_input_shape, dtype='float32')

        # Input for LSTM: 3D tensor: (samples_index, sequence_index, feature)
        global_model_input2 = Input(shape=(self.global_train_seq_length, global_input_size), dtype='float32')

        # Layer 1-1: ConvLSTM with kernel size (3, 3)
        global_predict_map = Bidirectional(ConvLSTM2D(self.global_num_filters, self.global_kernel_size,
                                                      padding='valid', activation='tanh',
                                                      recurrent_activation='hard_sigmoid', use_bias=True,
                                                      unit_forget_bias=True,
                                                      kernel_regularizer=l2(self.global_regularizer),
                                                      recurrent_regularizer=l2(self.global_regularizer),
                                                      bias_regularizer=l2(self.global_regularizer),
                                                      activity_regularizer=l2(self.global_regularizer),
                                                      dropout=self.global_cnn_dropout,
                                                      recurrent_dropout=self.global_r_dropout))(global_model_input)
        # global_predict_map = BatchNormalization()(global_predict_map)
        # global_predict_map = MaxPooling3D(pool_size=self.global_3d_pool_size)(global_predict_map)
        # global_predict_vec = Flatten()(global_predict_map)

        # Layer 1-2: ConvLSTM with kernel size (3, 3)
        # global_predict_map2 = Bidirectional(ConvLSTM2D(self.global_num_filters, self.global_kernel_size,
        #                                                padding='valid', activation='tanh',
        #                                                recurrent_activation='hard_sigmoid', use_bias=True,
        #                                                unit_forget_bias=True,
        #                                                kernel_regularizer=l2(self.global_regularizer),
        #                                                recurrent_regularizer=l2(self.global_regularizer),
        #                                                bias_regularizer=l2(self.global_regularizer),
        #                                                activity_regularizer=l2(self.global_regularizer),
        #                                                dropout=self.global_cnn_dropout,
        #                                                recurrent_dropout=self.global_r_dropout))(global_predict_map)
        global_predict_map2 = MaxPooling2D(pool_size=self.global_pool_size)(global_predict_map)
        global_predict_vec2 = Flatten()(global_predict_map2)

        # LSTM
        global_predict_map3 = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                                 beta_initializer="zero", gamma_initializer="one",
                                                 weights=None, gamma_regularizer=None,
                                                 momentum=0.99, axis=-1)(global_model_input2)
        global_predict_map3 = Bidirectional(LSTM(10, kernel_regularizer=l2(self.global_regularizer),
                                            recurrent_regularizer=l2(self.global_regularizer),
                                            bias_regularizer=l2(self.global_regularizer),
                                                 recurrent_dropout=0.8))(global_predict_map3)

        # Concatenation
        global_predict_map0 = concatenate([global_predict_vec2, global_predict_map3])

        # output layer
        global_output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                                 beta_initializer="zero", gamma_initializer="one",
                                                 weights=None, gamma_regularizer=None,
                                                 momentum=0.99, axis=-1)(global_predict_map0)

        global_output_layer = Dense(30, kernel_regularizer=l2(self.global_regularizer),
                                    bias_regularizer=l2(self.global_regularizer))(global_output_layer)
        global_output_layer = Dropout(0.5)(global_output_layer)

        global_output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                                 beta_initializer="zero", gamma_initializer="one",
                                                 weights=None, gamma_regularizer=None,
                                                 momentum=0.99, axis=-1)(global_output_layer)

        global_output_layer2 = Dense(output_size, kernel_regularizer=l2(self.global_regularizer),
                                     bias_regularizer=l2(self.global_regularizer), name='global_output')(global_output_layer)

        self.global_model = Model(inputs=[global_model_input, global_model_input2],
                                  outputs=global_output_layer2)

        if output_form[2] == 1:
            # Regression problem
            self.global_model.compile(loss=keras.losses.mean_squared_error, optimizer='nadam', metrics=['accuracy'])
        else:
            # Classification problem
            self.global_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='nadam', metrics=['accuracy'])
        #
        # ----- END: Global Models (LSTM & ConvLSTM) Definition --------------------------------------------------------
        #

        #
        # ----- START: Local Models (LSTM & ConvLSTM) Definition -------------------------------------------------------
        #
        # feature 'WIND_DIREC' has two dimension
        local_input_size = len(
            self.local_pollution_kind) + 1 if 'WIND_DIREC' in self.local_pollution_kind else len(
            self.local_pollution_kind)
        local_input_size = local_input_size + self.local_feature_kind_shift
        local_input_shape = (self.local_train_seq_length,
                             self.local_input_map_shape[0], self.local_input_map_shape[1], local_input_size)

        # Input for ConvLSTM: 5D tensor with shape: (samples_index, sequence_index, row, col, feature/channel)
        local_model_input = Input(shape=local_input_shape, dtype='float32')

        # Input for LSTM: 3D tensor: (samples_index, sequence_index, feature)
        local_model_input2 = Input(shape=(self.local_train_seq_length, local_input_size), dtype='float32')

        # Layer 1-1: ConvLSTM with kernel size (3, 3)
        local_predict_map = Bidirectional(ConvLSTM2D(self.local_num_filters, self.local_kernel_size,
                                                     padding='valid', activation='tanh',
                                                     recurrent_activation='hard_sigmoid', use_bias=True,
                                                     unit_forget_bias=True,
                                                     kernel_regularizer=l2(self.local_regularizer),
                                                     recurrent_regularizer=l2(self.local_regularizer),
                                                     bias_regularizer=l2(self.local_regularizer),
                                                     activity_regularizer=l2(self.local_regularizer),
                                                     dropout=self.local_cnn_dropout,
                                                     recurrent_dropout=self.local_r_dropout))(local_model_input)
        local_predict_map2 = MaxPooling2D(pool_size=self.local_pool_size)(local_predict_map)
        local_predict_vec2 = Flatten()(local_predict_map2)

        # LSTM
        local_predict_map3 = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                                beta_initializer="zero", gamma_initializer="one",
                                                weights=None, gamma_regularizer=None,
                                                momentum=0.99, axis=-1)(local_model_input2)
        local_predict_map3 = Bidirectional(LSTM(10, kernel_regularizer=l2(self.local_regularizer),
                                                recurrent_regularizer=l2(self.local_regularizer),
                                                bias_regularizer=l2(self.local_regularizer),
                                                recurrent_dropout=0.8))(local_predict_map3)

        # Concatenation
        local_predict_map0 = concatenate([local_predict_vec2, local_predict_map3, global_output_layer])

        # output layer
        local_output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                                beta_initializer="zero", gamma_initializer="one",
                                                weights=None, gamma_regularizer=None,
                                                momentum=0.99, axis=-1)(local_predict_map0)

        local_output_layer = Dense(30, kernel_regularizer=l2(self.local_regularizer),
                                   bias_regularizer=l2(self.local_regularizer))(local_output_layer)
        local_output_layer = Dropout(0.5)(local_output_layer)

        local_output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                                beta_initializer="zero", gamma_initializer="one",
                                                weights=None, gamma_regularizer=None,
                                                momentum=0.99, axis=-1)(local_output_layer)

        local_output_layer = Dense(output_size, kernel_regularizer=l2(self.local_regularizer),
                                   bias_regularizer=l2(self.local_regularizer), name='local_output')(local_output_layer)

        self.local_model = Model(inputs=[local_model_input, local_model_input2, global_model_input, global_model_input2],
                                 outputs=[local_output_layer, global_output_layer2])

        if output_form[2] == 1:
            # Regression problem
            self.local_model.compile(loss=keras.losses.mean_squared_error, optimizer='nadam', metrics=['accuracy'])#, loss_weights={'global_output': 0.2, 'local_output': 1})
        else:
            # Classification problem
            self.local_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='nadam',
                                     metrics=['accuracy'])
        #
        # ----- END: Local Models (LSTM & ConvLSTM) Definition ---------------------------------------------------------
        #

        # Specify the path where model will be saved.
        print('Finish building the forecast model...')


    #
    # Set label normalization parameters
    #
    def set_norm(self, norm_params):
        try:
            global_y_norm = norm_params['global']
            self.global_mean_y_train = global_y_norm['mean']
            self.global_std_y_train = global_y_norm['std']
        except Exception as e:
            print('[hybrid_forecast_model]', e)
            print('[hybrid_forecast_model] no global norm params')

        try:
            local_y_norm = norm_params['local']
            self.local_mean_y_train = local_y_norm['mean']
            self.local_std_y_train = local_y_norm['std']
        except Exception as e:
            print('[hybrid_forecast_model]', e)
            print('[hybrid_forecast_model] no global norm params')


    #
    # Load Model: load global NN model, local NN model, XGB and Ensemble model
    #
    def load_model(self, global_model_path, local_model_path, xgb_model_path, ens_model_path):

        print('loading model ..')

        self.global_model.load_weights(global_model_path)
        self.local_model.load_weights(local_model_path)

        with open(xgb_model_path, 'rb') as fr:
            self.xgb_model = pickle.load(fr)

        with open(ens_model_path, 'rb') as fr:
            self.ensemble_model = pickle.load(fr)




    #
    # Model Inference:
    #
    def inference(self, x_test_xgb, x_test_global, x_test_global2, x_test_local, x_test_local2):

        xgb_predict = self.xgb_model.predict(x_test_xgb)
        global_nn_predict = self.global_model.predict([x_test_global, x_test_global2])
        local_nn_predict, global_in_local_nn_predict = self.local_model.predict([x_test_local, x_test_local2,
                                                                                 x_test_global, x_test_global2])
        final_predict = self.ensemble_model.predict(np.hstack((x_test_xgb, xgb_predict, global_nn_predict,
                                                               global_in_local_nn_predict, local_nn_predict)))
        norm_predict = self.local_mean_y_train[0] + self.local_std_y_train[0] * final_predict

        return norm_predict


    #
    # Model evaluate(2 output):
    #
    def evaluate(self, x_test_xgb, x_test_global, x_test_global2, x_test_local, x_test_local2):

        xgb_predict = self.xgb_model.predict(x_test_xgb)
        global_nn_predict = self.global_model.predict([x_test_global, x_test_global2])
        local_nn_predict, global_in_local_nn_predict = self.local_model.predict([x_test_local, x_test_local2,
                                                                                 x_test_global, x_test_global2])
        final_predict = self.ensemble_model.predict(np.hstack((x_test_xgb, xgb_predict, global_nn_predict,
                                                               global_in_local_nn_predict, local_nn_predict)))
        norm_predict = self.local_mean_y_train[0] + self.local_std_y_train[0] * final_predict
        norm_global_predict = self.global_mean_y_train[0] + self.global_mean_y_train[0] * global_nn_predict

        return norm_predict, norm_global_predict


    #
    # Model Training:
    #
    def train(self, x_train_xgb,
              x_train_global, x_train_global2, x_train_local, x_train_local2,
              y_train_global, y_train_local,
              x_test_global, x_test_global2, x_test_local, x_test_local2,
              y_test_global, y_test_local,
              global_model_path, local_model_path, model_xgb_path, model_ensemble_path,
              x_full=None, x3_full=None,
              x_valid_full=None, x_valid3_full=None,
              y_full=None, y_valid_full=None):

        print("Train NN ...")
        start_time = time.time()

        #
        # ------ START: Train global model -----------------------------------------------------------------------------
        # ---- Pre-train on FULL EPA DATA ----
        # pretrain == 1 if pre-train data exist
        pretrain = 1
        if x_full is None:
            pretrain = 0
        if x3_full is None:
            pretrain = 0
        if x_valid_full is None:
            pretrain = 0
        if x_valid3_full is None:
            pretrain = 0
        if y_full is None:
            pretrain = 0
        if y_valid_full is None:
            pretrain = 0

        if not self.is_global_pretrained and pretrain == 1:
            self.global_model.fit(x=[x_full,
                                     x3_full],
                                  y=y_full,
                                  batch_size=self.global_batch_size,
                                  epochs=self.global_epoch,
                                  validation_data=([x_valid_full,
                                                    x_valid3_full],
                                                   y_valid_full),
                                  shuffle=True,
                                  callbacks=[EarlyStopping(monitor='val_loss', min_delta=0,
                                                           patience=6, verbose=0, mode='auto'),
                                             ModelCheckpoint(global_model_path, monitor='val_loss', verbose=0,
                                                             save_best_only=True, save_weights_only=True,
                                                             mode='auto', period=1)])
            # self.global_model.save_weights(global_model_path, overwrite=True)

        # ------ END: Train global model -------------------------------------------------------------------------------

        #
        # ------ START: Train local model ------------------------------------------------------------------------------
        if not self.is_local_pretrained:
            self.local_model.fit(x=[x_train_local,
                                    x_train_local2,
                                    x_train_global,
                                    x_train_global2],
                                 y=[y_train_local, y_train_global],
                                 batch_size=self.local_batch_size,
                                 epochs=self.local_epoch,
                                 validation_data=([x_test_local,
                                                   x_test_local2,
                                                   x_test_global,
                                                   x_test_global2],
                                                  [y_test_local, y_test_global]),
                                 shuffle=True,
                                 callbacks=[EarlyStopping(monitor='val_loss', min_delta=0,
                                                          patience=6, verbose=0, mode='auto'),
                                            ModelCheckpoint(local_model_path, monitor='val_loss', verbose=0,
                                                            save_best_only=True, save_weights_only=True,
                                                            mode='auto', period=1)])
            # self.local_model.save_weights(local_model_path, overwrite=True)
        # ------ END: Train local model --------------------------------------------------------------------------------

        #
        # ------ START: Train XGB model --------------------------------------------------------------------------------
        print('Train XGB ...')
        self.xgb_model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear', random_state=1234)).fit(x_train_xgb, y_train_local)

        with open(model_xgb_path, 'wb') as fw:
            pickle.dump(self.xgb_model, fw)
        # ------ END: Train XGB model ----------------------------------------------------------------------------------

        #
        # ------ START: Train ensemble model ---------------------------------------------------------------------------
        print('Stacking ...')
        xgb_predict = self.xgb_model.predict(x_train_xgb).reshape(len(x_train_xgb), 1)
        global_nn_predict = self.global_model.predict([x_train_global, x_train_global2])
        local_nn_predict, global_in_local_nn_predict = self.local_model.predict([x_train_local, x_train_local2, x_train_global, x_train_global2])
        x_train_ensemble = np.hstack((x_train_xgb, xgb_predict, global_nn_predict, global_in_local_nn_predict, local_nn_predict))
        self.ensemble_model = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:linear', random_state=1234)).fit(x_train_ensemble, y_train_local)

        with open(model_ensemble_path, 'wb') as fw:
            pickle.dump(self.ensemble_model, fw)
        # ------ END: Train ensemble model -----------------------------------------------------------------------------

        final_time = time.time()
        time_spent_printer(start_time, final_time)

    def train_global_only(self,
                          global_model_path,
                          x_full, x3_full,
                          x_valid_full, x_valid3_full,
                          y_full, y_valid_full):

        print("Train global only ...")
        start_time = time.time()
        self.global_model.fit(x=[x_full,
                                 x3_full],
                              y=y_full,
                              batch_size=self.global_batch_size,
                              epochs=self.global_epoch,
                              validation_data=([x_valid_full,
                                                x_valid3_full],
                                               y_valid_full),
                              shuffle=True,
                              callbacks=[EarlyStopping(monitor='val_loss', min_delta=0,
                                                       patience=6, verbose=0, mode='auto'),
                                         ModelCheckpoint(global_model_path, monitor='val_loss', verbose=0,
                                                         save_best_only=True, save_weights_only=True,
                                                         mode='auto', period=1)])
        # self.global_model.save_weights(global_model_path, overwrite=True)
        final_time = time.time()
        time_spent_printer(start_time, final_time)


