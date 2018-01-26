import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from datetime import date, timedelta
import gc

def load_data():
    # df_train = pd.read_feather('train_after1608_raw')
    df_train = pd.read_csv('train.csv', usecols=[1, 2, 3, 4, 5], dtype={'onpromotion': bool},
                           converters={'unit_sales': lambda u: np.log1p(float(u)) if float(u) > 0 else 0},
                           parse_dates=["date"])
    df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                          parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])

    # subset data
    df_2017 = df_train.loc[df_train.date>=pd.datetime(2016,1,1)]

    # promo
    promo_2017_train = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["onpromotion"]].unstack(
        level=-1).fillna(False)
    promo_2017_train.columns = promo_2017_train.columns.get_level_values(1)
    promo_2017_test = df_test[["onpromotion"]].unstack(level=-1).fillna(False)
    promo_2017_test.columns = promo_2017_test.columns.get_level_values(1)
    promo_2017_test = promo_2017_test.reindex(promo_2017_train.index).fillna(False)
    promo_2017 = pd.concat([promo_2017_train, promo_2017_test], axis=1)
    del promo_2017_test, promo_2017_train

    df_2017 = df_2017.set_index(
    ["store_nbr", "item_nbr", "date"])[["unit_sales"]].unstack(
        level=-1).fillna(0)
    df_2017.columns = df_2017.columns.get_level_values(1)

    # items
    items = pd.read_csv("items.csv").set_index("item_nbr")
    stores = pd.read_csv("stores.csv").set_index("store_nbr")
    # items = items.reindex(df_2017.index.get_level_values(1))

    return df_2017, promo_2017, items, stores

def save_unstack(df, promo, filename):
    df_name, promo_name = 'df_' + filename + '_raw', 'promo_' + filename + '_raw'
    df.columns = df.columns.astype('str')
    df.reset_index().to_feather(df_name)
    promo.columns = promo.columns.astype('str')
    promo.reset_index().to_feather(promo_name)

def load_unstack(filename):
    df_name, promo_name = 'df_' + filename + '_raw', 'promo_' + filename + '_raw'
    df_2017 = pd.read_feather(df_name).set_index(['store_nbr','item_nbr'])
    df_2017.columns = pd.to_datetime(df_2017.columns)
    promo_2017 = pd.read_feather(promo_name).set_index(['store_nbr','item_nbr'])
    promo_2017.columns = pd.to_datetime(promo_2017.columns)
    items = pd.read_csv("items.csv").set_index("item_nbr")
    stores = pd.read_csv("stores.csv").set_index("store_nbr")

    return df_2017, promo_2017, items, stores

# Create validation and test data
def create_dataset(df, promo_df, items, stores, timesteps, first_pred_start, is_train=True, aux_as_tensor=False, reshape_output=0):
    encoder = LabelEncoder()
    items_reindex = items.reindex(df.index.get_level_values(1))
    item_family = encoder.fit_transform(items_reindex['family'].values)
    item_class = encoder.fit_transform(items_reindex['class'].values)
    item_perish = items_reindex['perishable'].values

    stores_reindex = stores.reindex(df.index.get_level_values(0))
    store_nbr = df.reset_index().store_nbr.values - 1
    store_cluster = stores_reindex['cluster'].values - 1
    store_type = encoder.fit_transform(stores_reindex['type'].values)

    # item_mean_df = df.groupby('item_nbr').mean().reindex(df.index.get_level_values(1))
    item_group_mean = df.groupby('item_nbr').mean()
    store_group_mean = df.groupby('store_nbr').mean()
    # store_family_group_mean = df.join(items['family']).groupby(['store_nbr', 'family']).transform('mean')
    # store_family_group_mean.index = df.index

    cat_features = np.stack([item_family, item_class, item_perish, store_nbr, store_cluster, store_type], axis=1)

    return create_dataset_part(df, promo_df, cat_features, item_group_mean, store_group_mean, timesteps, first_pred_start, reshape_output, aux_as_tensor, is_train)

def train_generator(df, promo_df, items, stores, timesteps, first_pred_start,
    n_range=1, day_skip=7, is_train=True, batch_size=2000, aux_as_tensor=False, reshape_output=0, first_pred_start_2016=None):
    encoder = LabelEncoder()
    items_reindex = items.reindex(df.index.get_level_values(1))
    item_family = encoder.fit_transform(items_reindex['family'].values)
    item_class = encoder.fit_transform(items_reindex['class'].values)
    item_perish = items_reindex['perishable'].values

    stores_reindex = stores.reindex(df.index.get_level_values(0))
    store_nbr = df.reset_index().store_nbr.values - 1
    store_cluster = stores_reindex['cluster'].values - 1
    store_type = encoder.fit_transform(stores_reindex['type'].values)

    # item_mean_df = df.groupby('item_nbr').mean().reindex(df.index.get_level_values(1))
    item_group_mean = df.groupby('item_nbr').mean()
    store_group_mean = df.groupby('store_nbr').mean()
    # store_family_group_mean = df.join(items['family']).groupby(['store_nbr', 'family']).transform('mean')
    # store_family_group_mean.index = df.index

    cat_features = np.stack([item_family, item_class, item_perish, store_nbr, store_cluster, store_type], axis=1)

    while 1:
        date_part = np.random.permutation(range(n_range))
        if first_pred_start_2016 is not None:
            range_diff = (first_pred_start - first_pred_start_2016).days / day_skip
            date_part = np.concat([date_part, np.random.permutation(range(range_diff, int(n_range/2) + range_diff))])

        for i in date_part:
            keep_idx = np.random.permutation(df.shape[0])[:batch_size]
            df_tmp = df.iloc[keep_idx,:]
            promo_df_tmp = promo_df.iloc[keep_idx,:]
            cat_features_tmp = cat_features[keep_idx]
            # item_mean_tmp = item_mean_df.iloc[keep_idx, :]

            pred_start = first_pred_start - timedelta(days=int(day_skip*i))

            # Generate a batch of random subset data. All data in the same batch are in the same period.
            yield create_dataset_part(df_tmp, promo_df_tmp, cat_features_tmp, item_group_mean, store_group_mean, timesteps, pred_start, reshape_output, aux_as_tensor, True)

            gc.collect()

def create_dataset_part(df, promo_df, cat_features, item_group_mean, store_group_mean, timesteps, pred_start, reshape_output, aux_as_tensor, is_train, weight=False):

    item_mean_df = item_group_mean.reindex(df.index.get_level_values(1))
    store_mean_df = store_group_mean.reindex(df.index.get_level_values(0))
    # store_family_mean_df = store_family_group_mean.reindex(df.index)

    X, y = create_xy_span(df, pred_start, timesteps, is_train)
    is0 = (X==0).astype('uint8')
    promo = promo_df[pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)].values
    weekday = np.tile([d.weekday() for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
                          (X.shape[0],1))
    dom = np.tile([d.day-1 for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
                          (X.shape[0],1))
    item_mean, _ = create_xy_span(item_mean_df, pred_start, timesteps, False)
    store_mean, _ = create_xy_span(store_mean_df, pred_start, timesteps, False)
    # store_family_mean, _ = create_xy_span(store_family_mean_df, pred_start, timesteps, False)
    # month_tmp = np.tile([d.month-1 for d in pd.date_range(pred_start-timedelta(days=timesteps), periods=timesteps+16)],
    #                       (X_tmp.shape[0],1))
    yearAgo, _ = create_xy_span(df, pred_start-timedelta(days=365), timesteps+16, False)
    quarterAgo, _ = create_xy_span(df, pred_start-timedelta(days=91), timesteps+16, False)

    if reshape_output>0:
        X = X.reshape(-1, timesteps, 1)
    if reshape_output>1:
        is0 = is0.reshape(-1, timesteps, 1)
        promo = promo.reshape(-1, timesteps+16, 1)
        yearAgo = yearAgo.reshape(-1, timesteps+16, 1)
        quarterAgo = quarterAgo.reshape(-1, timesteps+16, 1)
        item_mean = item_mean.reshape(-1, timesteps, 1)
        store_mean = store_mean.reshape(-1, timesteps, 1)
        # store_family_mean = store_family_mean.reshape(-1, timesteps, 1)

    w = (cat_features[:, 2] * 0.25 + 1) / (cat_features[:, 2] * 0.25 + 1).mean()

    cat_features = np.tile(cat_features[:, None, :], (1, timesteps+16, 1)) if aux_as_tensor else cat_features

    # Use when only 6th-16th days (private periods) are in the training output
    # if is_train: y = y[:, 5:]

    if weight: return ([X, is0, promo, yearAgo, quarterAgo, weekday, dom, cat_features, item_mean, store_mean], y, w)
    else: return ([X, is0, promo, yearAgo, quarterAgo, weekday, dom, cat_features, item_mean, store_mean], y)

def create_xy_span(df, pred_start, timesteps, is_train=True, shift_range=0):
    X = df[pd.date_range(pred_start-timedelta(days=timesteps), pred_start-timedelta(days=1))].values
    if is_train: y = df[pd.date_range(pred_start, periods=16)].values
    else: y = None
    return X, y

# Not used in the final model
def random_shift_slice(mat, start_col, timesteps, shift_range):
    shift = np.random.randint(shift_range+1, size=(mat.shape[0],1))
    shift_window = np.tile(shift,(1,timesteps)) + np.tile(np.arange(start_col, start_col+timesteps),(mat.shape[0],1))
    rows = np.arange(mat.shape[0])
    rows = rows[:,None]
    columns = shift_window
    return mat[rows, columns]

# Calculate RMSE scores for all 16 days, first 5 days (fror public LB) and 6th-16th days (for private LB) 
def cal_score(Ytrue, Yfit):
	print([metrics.mean_squared_error(Ytrue, Yfit), 
	metrics.mean_squared_error(Ytrue[:,:5], Yfit[:,:5]),
	metrics.mean_squared_error(Ytrue[:,5:], Yfit[:,5:])])

# Create submission file
def make_submission(df_index, test_pred, filename):
	df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
	df_preds = pd.DataFrame(
	    test_pred, index=df_index,
	    columns=pd.date_range("2017-08-16", periods=16)
	).stack().to_frame("unit_sales")
	df_preds.index.set_names(["store_nbr", "item_nbr", "date"], inplace=True)

	submission = df_test[["id"]].join(df_preds, how="left").fillna(0)
	submission["unit_sales"] = np.clip(np.expm1(submission["unit_sales"]), 0, 1000)
	submission.to_csv(filename, float_format='%.4f', index=None)

# Thank you for watching! :)