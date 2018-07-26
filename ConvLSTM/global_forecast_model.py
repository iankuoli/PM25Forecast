import keras
from keras.layers import Input, LSTM, MaxPooling2D, concatenate, Bidirectional, Dense, Dropout, Flatten, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping
import xgboost as xgb

import pickle
import time

from utility.Utilities import *


class GlobalForecastModel:

    #
    # Initialize forecast model and declare LSTM & ConvLSTM model structures
    #
    def __init__(self, pollution_kind, target_kind, target_site, feature_kind_shift,
                 train_seq_seg, hyper_params, input_map_shape, output_form):

        self.pollution_kind = pollution_kind
        self.target_kind = target_kind
        self.target_site = target_site
        self.feature_kind_shift = feature_kind_shift
        self.train_seq_seg = train_seq_seg
        self.train_seq_length = int(self.train_seq_seg[0][0] / self.train_seq_seg[0][1])

        self.mean_y_train = np.zeros(len(self.target_kind))
        self.std_y_train = np.zeros(len(self.target_kind))

        self.num_filters = hyper_params['num_filters']
        self.kernel_size = hyper_params['kernel_size']
        self.regularizer = hyper_params['regularizer']
        self.cnn_dropout = hyper_params['cnn_dropout']
        self.r_dropout = hyper_params['r_dropout']
        self.pool_size = hyper_params['pool_size']
        self.epoch = hyper_params['epoch']
        self.batch_size = hyper_params['batch_size']
        self.interval_hours = hyper_params['interval_hours']
        self.input_map_shape = input_map_shape

        self.forecast = np.zeros(len(self.interval_hours))

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
        # ----- START: LSTM & ConvLSTM Models Definition ---------------------------------------------------------------
        #
        # feature 'WIND_DIREC' has two dimension
        input_size = len(self.pollution_kind) + 1 if 'WIND_DIREC' in self.pollution_kind else len(self.pollution_kind)
        input_size = input_size + self.feature_kind_shift
        input_shape = (self.train_seq_length, self.input_map_shape[0], self.input_map_shape[1], input_size)

        # Input for ConvLSTM: 5D tensor with shape: (samples_index, sequence_index, row, col, feature/channel)
        model_input = Input(shape=input_shape, dtype='float32')

        # Input for DNN: 2D matrix
        #model_input2 = Input(shape=(self.train_seq_length * input_size,), dtype='float32')

        # Input for LSTM: 3D tensor: (samples_index, sequence_index, feature)
        model_input3 = Input(shape=(self.train_seq_length, input_size), dtype='float32')

        # Layer 1-1: ConvLSTM with kernel size (3, 3)
        predict_map = Bidirectional(ConvLSTM2D(self.num_filters, self.kernel_size, padding='valid', activation='tanh',
                                               recurrent_activation='hard_sigmoid', use_bias=True,
                                               unit_forget_bias=True,
                                               kernel_regularizer=l2(self.regularizer),
                                               recurrent_regularizer=l2(self.regularizer),
                                               bias_regularizer=l2(self.regularizer),
                                               activity_regularizer=l2(self.regularizer),
                                               dropout=self.cnn_dropout, recurrent_dropout=self.r_dropout))(model_input)
        predict_map = MaxPooling2D(pool_size=self.pool_size)(predict_map)
        predict_vec = Flatten()(predict_map)

        # Layer 1-2: ConvLSTM with kernel size (3, 3)
        predict_map2 = Bidirectional(ConvLSTM2D(self.num_filters, self.kernel_size, padding='valid', activation='tanh',
                                                recurrent_activation='hard_sigmoid', use_bias=True,
                                                unit_forget_bias=True,
                                                kernel_regularizer=l2(self.regularizer),
                                                recurrent_regularizer=l2(self.regularizer),
                                                bias_regularizer=l2(self.regularizer),
                                                activity_regularizer=l2(self.regularizer),
                                                dropout=self.cnn_dropout, recurrent_dropout=self.r_dropout))(predict_map)
        predict_map2 = MaxPooling2D(pool_size=self.pool_size)(predict_map2)
        predict_vec2 = Flatten()(predict_map2)

        # LSTM
        predict_map3 = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                          beta_initializer="zero", gamma_initializer="one",
                                          weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(model_input3)
        predict_map3 = Bidirectional(LSTM(10, kernel_regularizer=l2(self.regularizer),
                                          recurrent_regularizer=l2(self.regularizer),
                                          bias_regularizer=l2(self.regularizer), recurrent_dropout=0.8))(predict_map3)

        # Concatenation
        predict_map0 = concatenate([predict_vec, predict_vec2, predict_map3])
        # predict_map0 = concatenate([predict_vec, predict_vec2, predict_map4, model_input2])

        # output layer
        output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                          beta_initializer="zero", gamma_initializer="one",
                                          weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(predict_map0)

        output_layer = Dense(30, kernel_regularizer=l2(self.regularizer),
                             bias_regularizer=l2(self.regularizer))(output_layer)
        output_layer = Dropout(0.5)(output_layer)

        output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                          beta_initializer="zero", gamma_initializer="one",
                                          weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(output_layer)

        output_layer = Dense(output_size, kernel_regularizer=l2(self.regularizer),
                             bias_regularizer=l2(self.regularizer))(output_layer)

        self.forecast_model = Model(inputs=[model_input, model_input3], outputs=output_layer)

        if output_form[2] == 1:
            self.forecast_model.compile(loss=keras.losses.mean_squared_error,
                                        optimizer='nadam', metrics=['accuracy'])
        else:
            self.forecast_model.compile(loss=keras.losses.categorical_crossentropy,
                                        optimizer='nadam', metrics=['accuracy'])
        #
        # ----- END: LSTM & ConvLSTM Models Definition -----------------------------------------------------------------
        #

        # Specify the path where model will be saved.
        print('Finish building the forecast model...')

    #
    # Load Model: load LSTM, ConvLSTM, XGB and Ensemble model
    #
    def load_model(self, nn_model_path, xgb_model_path, ens_model_path):

        print('loading model ..')
        self.forecast_model.load_weights(nn_model_path)

        with open(xgb_model_path, 'rb') as fr:
            self.xgb_model = pickle.load(fr)

        with open(ens_model_path, 'rb') as fr:
            self.ensemble_model = pickle.load(fr)

    #
    # Model Inference:
    #
    def inference(self, x_test_xgb, x_test_1, x_test_2, x_test_3):

        xgb_predict = self.xgb_model.predict(x_test_xgb)
        nn_predict = self.forecast_model.predict([x_test_1, x_test_2, x_test_3])
        final_predict = self.ensemble_model.predict(np.hstack((x_test_xgb, xgb_predict, nn_predict)))
        norm_predict = self.mean_y_train[0] + self.std_y_train[0] * final_predict

        return norm_predict

    #
    # Model Training:
    #
    def train(self, x_train_xgb, x_train_1, x_train_2, x_train_3, y_train, x_test_1, x_test_2, x_test_3, y_test,
              model_nn_path, model_xgb_path, model_ensemble_path):

        print("Train NN ...")
        start_time = time.time()

        #
        # ------ START: Train NN model ---------------------------------------------------------------------------------
        #
        self.forecast_model.fit(x=[x_train_1, x_train_2, x_train_3],
                                y=y_train,
                                batch_size=self.batch_size,
                                epochs=self.epoch,
                                validation_data=([x_test_1, x_test_2, x_test_3], y_test),
                                shuffle=True,
                                callbacks=[EarlyStopping(monitor='val_loss', min_delta=0,
                                                         patience=3, verbose=0, mode='auto'),
                                           ModelCheckpoint(model_nn_path, monitor='val_loss', verbose=0,
                                                           save_best_only=True, save_weights_only=True,
                                                           mode='auto', period=1)])
        self.forecast_model.save_weights(model_nn_path, overwrite=True)
        #
        # ------ END: Train NN model -----------------------------------------------------------------------------------

        #
        # ------ START: Train XGB model --------------------------------------------------------------------------------
        #
        print('Train XGB ...')
        self.xgb_model = xgb.XGBRegressor().fit(x_train_xgb, y_train[:, 0])

        with open(model_xgb_path, 'wb') as fw:
            pickle.dump(self.xgb_model, fw)
        #
        # ------ END: Train XGB model ----------------------------------------------------------------------------------

        #
        # ------ START: Train ensemble model ---------------------------------------------------------------------------
        #
        print('Stacking ...')
        xgb_predict = self.xgb_model.predict(x_train_xgb).reshape(len(x_train_xgb), 1)
        nn_predict = self.forecast_model.predict([x_train_1, x_train_2, x_train_3])
        x_train_ensemble = np.hstack((x_train_xgb, xgb_predict, nn_predict))
        self.ensemble_model = xgb.XGBRegressor().fit(x_train_ensemble, y_train[:, 0])

        with open(model_ensemble_path, 'wb') as fw:
            pickle.dump(self.ensemble_model, fw)
        #
        # ------ END: Train ensemble model -----------------------------------------------------------------------------

        final_time = time.time()
        time_spent_printer(start_time, final_time)