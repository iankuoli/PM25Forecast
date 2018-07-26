import time

import keras
from keras.layers import Input, Conv2D, LSTM, MaxPooling2D, concatenate, Bidirectional, Activation, TimeDistributed, Dense, Dropout, Flatten, ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, EarlyStopping

from reader import read_data_map, construct_time_map2
from missing_value_processer import missing_check
from config import root, dataset_path, site_map2
from Utilities import *


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

# Define the output form
output_form = (0, 80, 1, len(target_kind))      # Regression (min, max, num_of_slice, num_of_obj)
# output_form = (0, 150, 50, 1)                 # Classification (min, max, num_of_slice, num_of_obj)
if output_form[2] > 1:
    mean_form = [i for i in range(output_form[0], output_form[1], int(output_form[1]/output_form[2]))]
    mean_form = [i+int(output_form[1]/output_form[2])/2 for i in mean_form]
output_size = output_form[2] * output_form[3]

# Define target site and its adjacent map for prediction
pollution_site_map2 = site_map2()
target_site = pollution_site_map2['古亭']
center_i = int(target_site.shape[0]/2)
center_j = int(target_site.shape[1]/2)
local = target_site.local
city = target_site.city
target_site_name = target_site.site_name
site_list = list(target_site.adj_map.keys())  # ['士林', '中山', '松山', '汐止', '板橋', '萬華', '古亭', '土城', '新店']

# Define pre-processing parameters
feature_kind_shift = 6  # 'day of year', 'day of week' and 'time of day' respectively are represented by two dimensions

# Define model parameters
regularizer = 1e-7
batch_size = 256
num_filters = 16
kernel_size = (3, 3)
kernel_size2 = (2, 2)
pool_size = (2, 2)
epoch = 50
r_dropout = 0.5
cnn_dropout = 0.5
dnn_dropout = 0.5
train_seq_seg = [(36, 1)] #[(12, 1), (24, 2), (48, 3), (72, 6)]
train_seq_length = int(train_seq_seg[0][0] / train_seq_seg[0][1])
for seg_idx in range(1, len(train_seq_seg)):
    train_seq_length += int((train_seq_seg[seg_idx][0] - train_seq_seg[seg_idx-1][0]) / train_seq_seg[seg_idx][1])
# train_seq_length = 48

#
# ----- END: Parameters Declaration ------------------------------------------------------------------------------------
#


#
# ----- START: Year Processing -----------------------------------------------------------------------------------------
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
# ----- END: Year Processing -------------------------------------------------------------------------------------------
#


#
# ----- START: Data Loading --------------------------------------------------------------------------------------------
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
X_train = read_data_map(path=data_path, site=target_site, feature_selection=pollution_kind,
                        date_range=np.atleast_1d(training_year), beginning=training_duration[0],
                        finish=training_duration[-1], update=data_update)
X_train = missing_check(X_train)
Y_train = np.array(X_train)[:, center_i, center_j, [6 + pollution_kind.index(i) for i in target_kind]]

# Load testing data, where: size(X_test) = (data_size, map_l, map_w, map_h), not sequentialized yet.
print('Preparing testing dataset ..')
X_test = read_data_map(path=data_path, site=target_site, feature_selection=pollution_kind,
                       date_range=np.atleast_1d(testing_year), beginning=testing_duration[0],
                       finish=testing_duration[-1], update=data_update)
X_test = missing_check(X_test)
Y_test = np.array(X_test)[:, center_i, center_j, [6 + pollution_kind.index(i) for i in target_kind]]

# Set end time of data loading
final_time = time.time()
print('Reading data .. ok, ', end='')
time_spent_printer(start_time, final_time)

#
# ----- END: Data Loading ----------------------------------------------------------------------------------------------
#


#
# ----- START: Data Pre-processing -------------------------------------------------------------------------------------
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
    if not(Y_test[i, 1] > -10000):  # check missing or not, if Y_test[i] is missing, then this command will return True
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
# ----- END: Data Pre-processing ---------------------------------------------------------------------------------------
#


#
# ----- START: Data Partition ------------------------------------------------------------------------------------------
#

# Validation set
X_valid = X_train[-800:]
Y_valid = Y_train[-800:]

# Training set
X_train = X_train[:-800]
Y_train = Y_train[:-800]

# Generate model_input2 from model_input
X_train2 = X_train[:, :, 2, 2, :].reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[4])
X_test2 = X_test[:, :, 2, 2, :].reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[4])

# Generate model_input3 from model_input
X_train3 = X_train[:, :, 2, 2, :]
X_test3 = X_test[:, :, 2, 2, :]
# X_train3 = np.zeros(shape=(X_train.shape[0], X_train.shape[1], len(target_site.adj_map) * X_train.shape[4]),
#                     dtype='float32')
# X_test3 = np.zeros(shape=(X_test.shape[0], X_test.shape[1], len(target_site.adj_map) * X_test.shape[4]),
#                    dtype='float32')
# indx = 0
# for i in target_site.adj_map:
#     (x, y) = target_site.adj_map[i]
#     X_train3[:, :, indx*X_test.shape[4]:(indx+1)*X_test.shape[4]] = X_train[:, :, x, y, :]
#     X_test3[:, :, indx*X_test.shape[4]:(indx+1)*X_test.shape[4]] = X_test[:, :, x, y, :]
#     indx += 1

print('Take 800 data to validation set')

#
# ----- END: Data Partition --------------------------------------------------------------------------------------------
#


#
# ----- START: Model Definition ----------------------------------------------------------------------------------------
#

# Specify the path where model will be saved.
model_saved_path = ("rnn_%s_training_%s_m%s_to_%s_m%s_interval_%s_%s"
                    % (target_site_name, training_year[0], training_start_point, training_year[-1], training_end_point,
                       interval_hours, target_kind))
print(model_saved_path)

print('Build rnn model...')
start_time = time.time()

# feature 'WIND_DIREC' has two dimension
input_size = len(pollution_kind)+1 if 'WIND_DIREC' in pollution_kind else len(pollution_kind)
input_size = input_size + feature_kind_shift
input_shape = (train_seq_length, 5, 5, input_size)

# Input for ConvLSTM: 5D tensor with shape: (samples_index, sequence_index, row, col, feature/channel)
model_input = Input(shape=input_shape, dtype='float32')

# Input for DNN: 2D matrix
model_input2 = Input(shape=(train_seq_length*input_size,), dtype='float32')

# Input for LSTM: 3D tensor: (samples_index, sequence_index, feature)
model_input3 = Input(shape=(train_seq_length, input_size), dtype='float32')
# model_input3 = Input(shape=(train_seq_length, len(target_site.adj_map) * input_size), dtype='float32')

# Layer 1-1: ConvLSTM with kernel size (3, 3)
predict_map = Bidirectional(ConvLSTM2D(num_filters, kernel_size, padding='valid', activation='tanh',
                                       recurrent_activation='hard_sigmoid', use_bias=True, unit_forget_bias=True,
                                       kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
                                       bias_regularizer=l2(regularizer), activity_regularizer=l2(regularizer),
                                       dropout=cnn_dropout, recurrent_dropout=r_dropout))(model_input)
predict_map = MaxPooling2D(pool_size=pool_size)(predict_map)
predict_map = Flatten()(predict_map)

# Layer 1-2: ConvLSTM with kernel size (2, 2)
predict_map2 = Bidirectional(ConvLSTM2D(num_filters, kernel_size2, padding='valid', activation='tanh',
                                        recurrent_activation='hard_sigmoid', use_bias=True, unit_forget_bias=True,
                                        kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
                                        bias_regularizer=l2(regularizer), activity_regularizer=l2(regularizer),
                                        dropout=cnn_dropout, recurrent_dropout=r_dropout))(model_input)
predict_map2 = MaxPooling2D(pool_size=pool_size)(predict_map2)
predict_map2 = Flatten()(predict_map2)

# Layer 1-3: ConvLSTM with kernel size (1, 1)
# predict_map3 = Bidirectional(ConvLSTM2D(num_filters, (4, 4), padding='valid', activation='tanh',
#                              recurrent_activation='hard_sigmoid', use_bias=True, unit_forget_bias=True,
#                              kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
#                              bias_regularizer=l2(regularizer), activity_regularizer=l2(regularizer),
#                              dropout=cnn_dropout, recurrent_dropout=r_dropout))(model_input)
# predict_map3 = MaxPooling2D(pool_size=pool_size)(predict_map3)
# predict_map3 = Flatten()(predict_map3)

# LSTM
predict_map4 = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                  beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(model_input3)
predict_map4 = Bidirectional(LSTM(10, kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
                                  bias_regularizer=l2(regularizer), recurrent_dropout=0.8))(predict_map4)

# CNN
# predict_map5 = TimeDistributed(Conv2D(num_filters, kernel_size, padding='valid', activation='relu', use_bias=True,
#                                       kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer),
#                                       activity_regularizer=l2(regularizer)))(model_input)
# predict_map5 = TimeDistributed(MaxPooling2D(pool_size=pool_size))(predict_map5)
# predict_map5 = TimeDistributed(Flatten())(predict_map5)
# predict_map5 = Bidirectional(LSTM(10, kernel_regularizer=l2(regularizer), recurrent_regularizer=l2(regularizer),
#                                   bias_regularizer=l2(regularizer), recurrent_dropout=0.8))(predict_map5)

# Concatenation
#predict_map0 = concatenate([predict_map, predict_map2, predict_map4, predict_map5, model_input2])
predict_map0 = concatenate([predict_map, predict_map2, predict_map4, model_input2])
#predict_map0 = concatenate([predict_map4, predict_map5, model_input2])

# output layer
output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                  beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(predict_map0)

output_layer = Dense(30, kernel_regularizer=l2(regularizer), bias_regularizer=l2(regularizer))(output_layer)
output_layer = Dropout(0.5)(output_layer)

output_layer = BatchNormalization(beta_regularizer=None, epsilon=0.001,
                                  beta_initializer="zero", gamma_initializer="one",
                                  weights=None, gamma_regularizer=None, momentum=0.99, axis=-1)(output_layer)

if output_form[2] == 1:
    output_layer = Dense(output_size, kernel_regularizer=l2(regularizer),
                         bias_regularizer=l2(regularizer))(output_layer)
else:
    output_layer = Dense(output_size, kernel_regularizer=l2(regularizer),
                         bias_regularizer=l2(regularizer), activation='softmax')(output_layer)

ConvLSTM_model = Model(inputs=[model_input, model_input2, model_input3], outputs=output_layer)

if output_form[2] == 1:
    ConvLSTM_model.compile(loss=keras.losses.mean_squared_error, optimizer='nadam', metrics=['accuracy'])
else:
    ConvLSTM_model.compile(loss=keras.losses.categorical_crossentropy, optimizer='nadam', metrics=['accuracy'])


final_time = time.time()
time_spent_printer(start_time, final_time)

#
# ----- END: Model Definition ------------------------------------------------------------------------------------------
#


#
# ----- START: Model Training ------------------------------------------------------------------------------------------
#

if is_training:
    print("Train...")
    start_time = time.time()

    ConvLSTM_model.fit(x=[X_train, X_train2, X_train3],
                       y=Y_train,
                       batch_size=batch_size,
                       epochs=epoch,
                       validation_data=([X_test, X_test2, X_test3], Y_test),
                       # validation_data=(X_valid, Y_valid),
                       shuffle=True,
                       callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto'),
                                  ModelCheckpoint(folder + model_saved_path, monitor='val_loss', verbose=0,
                                                  save_best_only=True, save_weights_only=True, mode='auto', period=1)])

    # Potentially save weights
    ConvLSTM_model.save_weights(folder + model_saved_path, overwrite=True)

    final_time = time.time()
    time_spent_printer(start_time, final_time)
    print('model saved: ', model_saved_path)

else:
    print('loading model ..')
    ConvLSTM_model.load_weights(folder + model_saved_path)

#
# ----- END: Model Training --------------------------------------------------------------------------------------------
#


#
# ----- START: Model Testing -------------------------------------------------------------------------------------------
#

ConvLSTM_pred = ConvLSTM_model.predict([X_test, X_test2, X_test3])
final_time = time.time()
time_spent_printer(start_time, final_time)

if output_form[2] == 1:
    # Regression: Normalize the dependent variable Y in the training dataset.
    pred = mean_y_train[0] + std_y_train[0] * ConvLSTM_pred[:, 0]
    tmp = np.atleast_2d(Y_test_original[:, 0])
    print('RMSE(rnn): %.5f' % (np.mean((np.atleast_2d(Y_test_original[:, 0])[0] - pred) ** 2, 0) ** 0.5))
else:
    # Classification: Use interval to slice the variable Y in the training dataset.
    pred = sum((ConvLSTM_pred * mean_form).T)
    print('RMSE(rnn): %.5f' % (np.mean((Y_test_original[:, 0] - pred)**2, 0)**0.5))
#
# ----- END: Model Testing ---------------------------------------------------------------------------------------------
#


filename = 'plot_PM25'
with open(root_path + '%s.ods' % filename, 'w') as fw:
    for j in Y_test_original[:, 0]: #Y_test[:, 0]:
        print('%f,' % j, file=fw, end="")
    fw.write('\n')
    for j in pred:
        print('%f,' % j, file=fw, end="")
    fw.write('\n')
    if output_form[2] == 1:
        print('RMSE(' + target_kind[0] + '): %.5f' % (np.mean((np.atleast_2d(Y_test_original[:, 0])[0] - pred) ** 2, 0) ** 0.5))
    else:
        print('RMSE(' + target_kind[0] + '): %.5f' % (np.mean((Y_test_original[:, 0] - pred) ** 2, 0) ** 0.5))

filename = 'plot_O3'
with open(root_path + '%s.ods' % filename, 'w') as fw:
    for j in Y_test[:, 1]:
        print('%f,' % j, file=fw, end="")
    fw.write('\n')
    for j in ConvLSTM_pred[:, 1]: #pred:
        print('%f,' % j, file=fw, end="")
    fw.write('\n')
    if output_form[2] == 1:
        print('RMSE(' + target_kind[1] + '): %.5f' % (np.mean((np.atleast_2d(Y_test_original[:, 1])[0] - pred) ** 2, 0) ** 0.5))
    else:
        print('RMSE(' + target_kind[1] + '): %.5f' % (np.mean((Y_test_original[:, 1] - pred) ** 2, 0) ** 0.5))

filename = 'plot_NOx'
with open( root_path + '%s.ods' % filename, 'w' ) as fw:
    for j in Y_test[:, 2]:
        print( '%f,' % j, file=fw, end="" )
    fw.write( '\n' )
    for j in ConvLSTM_pred[:, 2]:  # pred:
        print( '%f,' % j, file=fw, end="" )
    fw.write( '\n' )
    if output_form[2] == 1:
        print('RMSE(' + target_kind[2] + '): %.5f' % (np.mean((np.atleast_2d( Y_test_original[:, 2] )[0] - pred) ** 2, 0) ** 0.5))
    else:
        print('RMSE(' + target_kind[2] + '): %.5f' % (np.mean((Y_test_original[:, 2] - pred) ** 2, 0) ** 0.5))
