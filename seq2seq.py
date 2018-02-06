import os
import numpy as np
import pandas as pd
from datetime import date, timedelta
from sklearn import metrics
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
import keras
from keras.models import Sequential, Model
from keras.layers import *
from keras import optimizers
import gc

from Utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress tf warnings

timesteps = 365

df, promo_df, items, stores = load_unstack('all')

# data after 2015
df = df[pd.date_range(date(2014,6,1), date(2017,8,15))]
promo_df = promo_df[pd.date_range(date(2014,6,1), date(2017,8,31))]

promo_df = promo_df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df = df[df[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_df = promo_df.astype('int')

df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))
df = df.loc[df.index.get_level_values(1).isin(item_inter)]
promo_df = promo_df.loc[promo_df.index.get_level_values(1).isin(item_inter)]

df_index = df.index
del item_nbr_test, item_nbr_train, item_inter, df_test; gc.collect()

train_data = train_generator(df, promo_df, items, stores, timesteps, date(2017, 7, 9),
                                           n_range=380, day_skip=1, batch_size=1000, aux_as_tensor=True, reshape_output=2)
Xval, Yval = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 7, 26),
                                     aux_as_tensor=True, reshape_output=2)
Xtest, _ = create_dataset(df, promo_df, items, stores, timesteps, date(2017, 8, 16),
                                    aux_as_tensor=True, is_train=False, reshape_output=2)

w = (Xval[7][:, 0, 2] * 0.25 + 1) / (Xval[7][:, 0, 2] * 0.25 + 1).mean()

del df, promo_df; gc.collect()

# Note
# current best: add item_mean, dim: 50, all as tensor ~ 3500 (~3630 in new cv)
print('1*100, train on private 7, nrange 380 timestep 200, data 1000*1500 \n')

latent_dim = 100

# seq input
seq_in = Input(shape=(timesteps, 1))
is0_in = Input(shape=(timesteps, 1))
promo_in = Input(shape=(timesteps+16, 1))
yearAgo_in = Input(shape=(timesteps+16, 1))
quarterAgo_in = Input(shape=(timesteps+16, 1))
item_mean_in = Input(shape=(timesteps, 1))
store_mean_in = Input(shape=(timesteps, 1))
# store_family_mean_in = Input(shape=(timesteps, 1))
weekday_in = Input(shape=(timesteps+16,), dtype='uint8')
weekday_embed_encode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
# weekday_embed_decode = Embedding(7, 4, input_length=timesteps+16)(weekday_in)
dom_in = Input(shape=(timesteps+16,), dtype='uint8')
dom_embed_encode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# dom_embed_decode = Embedding(31, 4, input_length=timesteps+16)(dom_in)
# weekday_onehot = Lambda(K.one_hot, arguments={'num_classes': 7}, output_shape=(timesteps+16, 7))(weekday_in)

# aux input
cat_features = Input(shape=(timesteps+16, 6))
item_family = Lambda(lambda x: x[:, :, 0])(cat_features)
item_class = Lambda(lambda x: x[:, :, 1])(cat_features)
item_perish = Lambda(lambda x: x[:, :, 2])(cat_features)
store_nbr = Lambda(lambda x: x[:, :, 3])(cat_features)
store_cluster = Lambda(lambda x: x[:, :, 4])(cat_features)
store_type = Lambda(lambda x: x[:, :, 5])(cat_features)

# store_in = Input(shape=(timesteps+16,), dtype='uint8')
family_embed = Embedding(33, 8, input_length=timesteps+16)(item_family)
class_embed = Embedding(337, 8, input_length=timesteps+16)(item_class)
store_embed = Embedding(54, 8, input_length=timesteps+16)(store_nbr)
cluster_embed = Embedding(17, 3, input_length=timesteps+16)(store_cluster)
type_embed = Embedding(5, 2, input_length=timesteps+16)(store_type)

# Encoder
encode_slice = Lambda(lambda x: x[:, :timesteps, :])
encode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in, weekday_embed_encode,
                               family_embed, Reshape((timesteps+16,1))(item_perish), store_embed, cluster_embed, type_embed], axis=2)
encode_features = encode_slice(encode_features)

# conv_in = Conv1D(8, 5, padding='same')(concatenate([seq_in, encode_features], axis=2))
# conv_raw = concatenate([seq_in, encode_slice(quarterAgo_in), encode_slice(yearAgo_in), item_mean_in], axis=2)
# conv_in = Conv1D(8, 5, padding='same')(conv_raw)
conv_in = Conv1D(4, 5, padding='same')(seq_in)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=1)(seq_in)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=2)(conv_in_deep)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=4)(conv_in_deep)
# conv_in_deep = Conv1D(2, 2, padding='causal', dilation_rate=8)(conv_in_deep)
# conv_in_quarter = Conv1D(4, 5, padding='same')(encode_slice(quarterAgo_in))
# conv_in_year = Conv1D(4, 5, padding='same')(encode_slice(yearAgo_in))
# conv_in = concatenate([conv_in_seq, conv_in_deep, conv_in_quarter, conv_in_year])

x_encode = concatenate([seq_in, encode_features, conv_in, item_mean_in], axis=2)
                        # store_mean_in, is0_in, store_family_mean_in], axis=2)
# encoder1 = CuDNNGRU(latent_dim, return_state=True, return_sequences=True)
# encoder2 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# encoder3 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
encoder = CuDNNGRU(latent_dim, return_state=True)
print('Input dimension:', x_encode.shape)
_, h= encoder(x_encode)
# s1, h1 = encoder1(x_encode)
# s1 = Dropout(0.25)(s1)
# s2, h2 = encoder2(s1)
# _, h3 = encoder3(s2)

# Connector
h = Dense(latent_dim, activation='tanh')(h)
# h1 = Dense(latent_dim, activation='tanh')(h1)
# h2 = Dense(latent_dim, activation='tanh')(h2)

# Decoder
previous_x = Lambda(lambda x: x[:, -1, :])(seq_in)

decode_slice = Lambda(lambda x: x[:, timesteps:, :])
decode_features = concatenate([promo_in, yearAgo_in, quarterAgo_in, weekday_embed_encode,
                               family_embed, Reshape((timesteps+16,1))(item_perish), store_embed, cluster_embed, type_embed], axis=2)
decode_features = decode_slice(decode_features)

# decode_idx_train = np.tile(np.arange(16), (Xtrain.shape[0], 1))
# decode_idx_val = np.tile(np.arange(16), (Xval.shape[0], 1))
# decode_idx = Input(shape=(16,))
# decode_id_embed = Embedding(16, 4, input_length=16)(decode_idx)
# decode_features = concatenate([decode_features, decode_id_embed])

# aux_features = concatenate([dom_embed_decode, store_embed_decode, family_embed_decode], axis=2)
# aux_features = decode_slice(aux_features)

# decoder1 = CuDNNGRU(latent_dim, return_state=True, return_sequences=True)
# decoder2 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# decoder3 = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
decoder = CuDNNGRU(latent_dim, return_state=True, return_sequences=False)
# decoder_dense1 = Dense(128, activation='relu')
decoder_dense2 = Dense(1, activation='relu')
# dp = Dropout(0.25)
slice_at_t = Lambda(lambda x: tf.slice(x, [0,i,0], [-1,1,-1]))
for i in range(16):
    previous_x = Reshape((1,1))(previous_x)
    
    features_t = slice_at_t(decode_features)
    # aux_t = slice_at_t(aux_features)

    decode_input = concatenate([previous_x, features_t], axis=2)
    # output_x, h1 = decoder1(decode_input, initial_state=h1)
    # output_x = dp(output_x)
    # output_x, h2 = decoder2(output_x, initial_state=h2)
    # output_x, h3 = decoder3(output_x, initial_state=h3)
    output_x, h = decoder(decode_input, initial_state=h)
    # aux input
    # output_x = concatenate([output_x, aux_t], axis=2)
    # output_x = Flatten()(output_x)
    # decoder_dense1 = Dense(64, activation='relu')
    # output_x = decoder_dense1(output_x)
    # output_x = dp(output_x)
    output_x = decoder_dense2(output_x)

    # gather outputs
    if i == 0: decoder_outputs = output_x
    elif i > 0: decoder_outputs = concatenate([decoder_outputs, output_x])

    previous_x = output_x

model = Model([seq_in, is0_in, promo_in, yearAgo_in, quarterAgo_in, weekday_in, dom_in, cat_features, item_mean_in, store_mean_in], decoder_outputs)

# rms = optimizers.RMSprop(lr=0.002)
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit_generator(train_data, steps_per_epoch=1500, workers=5, use_multiprocessing=True, epochs=18, verbose=2,
                    validation_data=(Xval, Yval, w))

# val_pred = model.predict(Xval)
# cal_score(Yval, val_pred)

test_pred = model.predict(Xtest)
make_submission(df_index, test_pred, 'seq-private_only-7.csv')

# model.save('save_models/seq2seq_model-withput-promo-2')
