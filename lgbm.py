from datetime import date, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import gc

from Utils import *

df_2017, promo_2017, items = load_unstack('1617')

promo_2017 = promo_2017[df_2017[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
df_2017 = df_2017[df_2017[pd.date_range(date(2017,1,1), date(2017,8,15))].max(axis=1)>0]
promo_2017 = promo_2017.astype('int')
df_test = pd.read_csv("test.csv", usecols=[0, 1, 2, 3, 4], dtype={'onpromotion': bool},
                      parse_dates=["date"]).set_index(['store_nbr', 'item_nbr', 'date'])
item_nbr_test = df_test.index.get_level_values(1)
item_nbr_train = df_2017.index.get_level_values(1)
item_inter = list(set(item_nbr_train).intersection(set(item_nbr_test)))

df_2017 = df_2017.loc[df_2017.index.get_level_values(1).isin(item_inter)]
promo_2017 = promo_2017.loc[promo_2017.index.get_level_values(1).isin(item_inter)]


def get_timespan(df, dt, minus, periods, freq='D'):
    return df[pd.date_range(dt - timedelta(days=minus), periods=periods, freq=freq)]

def prepare_dataset(t2017, is_train=True, one_hot=False):
    X = pd.DataFrame({
        "day_1_2017": get_timespan(df_2017, t2017, 1, 1).values.ravel(),
        "day_2_2017": get_timespan(df_2017, t2017, 2, 1).values.ravel(),
        "day_3_2017": get_timespan(df_2017, t2017, 3, 1).values.ravel(),
#         "day_4_2017": get_timespan(df_2017, t2017, 4, 1).values.ravel(),
#         "day_5_2017": get_timespan(df_2017, t2017, 5, 1).values.ravel(),
#         "day_6_2017": get_timespan(df_2017, t2017, 6, 1).values.ravel(),
#         "day_7_2017": get_timespan(df_2017, t2017, 7, 1).values.ravel(),
#         "mean_3_2017": get_timespan(df_2017, t2017, 3, 3).mean(axis=1).values,
#         "std_7_2017": get_timespan(df_2017, t2017, 7, 7).std(axis=1).values,
#         "max_7_2017": get_timespan(df_2017, t2017, 7, 7).max(axis=1).values,
#         "median_7_2017": get_timespan(df_2017, t2017, 7, 7).median(axis=1).values,
#         "median_30_2017": get_timespan(df_2017, t2017, 30, 30).median(axis=1).values,
#         "median_140_2017": get_timespan(df_2017, t2017, 140, 140).median(axis=1).values,
        'promo_3_2017': get_timespan(promo_2017, t2017, 3, 3).sum(axis=1).values,
        "last_year_mean": get_timespan(df_2017, t2017, 365, 16).mean(axis=1).values,
        "last_year_count0": (get_timespan(df_2017, t2017, 365, 16)==0).sum(axis=1).values,
        "last_year_promo": get_timespan(promo_2017, t2017, 365, 16).sum(axis=1).values
    })
    
    for i in [7, 14, 21, 30, 60, 90, 140, 365]:
        X['mean_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).mean(axis=1).values
        X['median_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).mean(axis=1).values
        X['max_{}_2017'.format(i)] = get_timespan(df_2017, t2017, i, i).max(axis=1).values
        X['mean_{}_haspromo_2017'.format(i)] = get_timespan(df_2017, t2017, i, i)[get_timespan(promo_2017, t2017, i, i)==1].mean(axis=1).values
        X['mean_{}_nopromo_2017'.format(i)] = get_timespan(df_2017, t2017, i, i)[get_timespan(promo_2017, t2017, i, i)==0].mean(axis=1).values
        X['count0_{}_2017'.format(i)] = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).values
        X['promo_{}_2017'.format(i)] = get_timespan(promo_2017, t2017, i, i).sum(axis=1).values
        item_mean = get_timespan(df_2017, t2017, i, i).mean(axis=1).groupby('item_nbr').mean().to_frame('item_mean')
        X['item_{}_mean'.format(i)] = df_2017.join(item_mean)['item_mean'].values
        item_count0 = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).groupby('item_nbr').mean().to_frame('item_count0')
        X['item_{}_count0_mean'.format(i)] = df_2017.join(item_count0)['item_count0'].values
        store_mean = get_timespan(df_2017, t2017, i, i).mean(axis=1).groupby('store_nbr').mean().to_frame('store_mean')
        X['store_{}_mean'.format(i)] = df_2017.join(store_mean)['store_mean'].values
        store_count0 = (get_timespan(df_2017, t2017, i, i)==0).sum(axis=1).groupby('store_nbr').mean().to_frame('store_count0')
        X['store_{}_count0_mean'.format(i)] = df_2017.join(store_count0)['store_count0'].values
        
    for i in range(7):
        X['mean_4_dow{}'.format(i)] = get_timespan(df_2017, t2017, 28-i, 4, freq='7D').mean(axis=1).values
        X['mean_10_dow{}'.format(i)] = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).values
        X['count0_10_dow{}'.format(i)] = (get_timespan(df_2017, t2017, 70-i, 10)==0).sum(axis=1).values
        X['promo_10_dow{}'.format(i)] = get_timespan(promo_2017, t2017, 70-i, 10, freq='7D').sum(axis=1).values
        item_mean = get_timespan(df_2017, t2017, 70-i, 10, freq='7D').mean(axis=1).groupby('item_nbr').mean().to_frame('item_mean')
        X['item_mean_10_dow{}'.format(i)] = df_2017.join(item_mean)['item_mean'].values
        X['mean_20_dow{}'.format(i)] = get_timespan(df_2017, t2017, 140-i, 20, freq='7D').mean(axis=1).values
        
    for i in range(16):
        X["promo_{}".format(i)] = promo_2017[t2017 + timedelta(days=i)].values
    
    if one_hot:
        family_dummy = pd.get_dummies(df_2017.join(items)['family'], prefix='family')
        X = pd.concat([X, family_dummy.reset_index(drop=True)], axis=1)
        store_dummy = pd.get_dummies(df_2017.reset_index().store_nbr, prefix='store')
        X = pd.concat([X, store_dummy.reset_index(drop=True)], axis=1)
#         X['family_count'] = df_2017.join(items).groupby('family').count().iloc[:,0].values
#         X['store_count'] = df_2017.reset_index().groupby('family').count().iloc[:,0].values
    else:
        df_items = df_2017.join(items)
        df_stores = df_2017.join(stores)
        X['family'] = df_items['family'].astype('category').cat.codes.values
        X['perish'] = df_items['perishable'].values
        X['item_class'] = df_items['class'].values
        X['store_nbr'] = df_2017.reset_index().store_nbr.values
        X['store_cluster'] = df_stores['cluster'].values
        X['store_type'] = df_stores['type'].astype('category').cat.codes.values
#     X['item_nbr'] = df_2017.reset_index().item_nbr.values
#     X['item_mean'] = df_2017.join(item_mean)['item_mean']
#     X['store_mean'] = df_2017.join(store_mean)['store_mean']

#     store_promo_90_mean = get_timespan(promo_2017, t2017, 90, 90).sum(axis=1).groupby('store_nbr').mean().to_frame('store_promo_90_mean')
#     X['store_promo_90_mean'] = df_2017.join(store_promo_90_mean)['store_promo_90_mean'].values
#     item_promo_90_mean = get_timespan(promo_2017, t2017, 90, 90).sum(axis=1).groupby('item_nbr').mean().to_frame('item_promo_90_mean')
#     X['item_promo_90_mean'] = df_2017.join(item_promo_90_mean)['item_promo_90_mean'].values
    
    if is_train:
        y = df_2017[pd.date_range(t2017, periods=16)].values
        return X, y
    return X


print("Preparing dataset...")
X_l, y_l = [], []
t2017 = date(2017, 7, 5)
n_range = 14
for i in range(n_range):
    print(i, end='..')
    delta = timedelta(days=7 * i)
    X_tmp, y_tmp = prepare_dataset(t2017 - delta)
    X_l.append(X_tmp)
    y_l.append(y_tmp)
    
X_train = pd.concat(X_l, axis=0)
y_train = np.concatenate(y_l, axis=0)
del X_l, y_l
X_val, y_val = prepare_dataset(date(2017, 7, 26))
X_test = prepare_dataset(date(2017, 8, 16), is_train=False)

params = {
    'num_leaves': 31,
    'objective': 'regression',
    'min_data_in_leaf': 300,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 2,
    'metric': 'l2',
    'max_bin':128,
    'num_threads': 8
}

print("Training and predicting models...")
MAX_ROUNDS = 700
val_pred = []
test_pred = []
# best_rounds = []
cate_vars = ['family', 'perish', 'store_nbr', 'store_cluster', 'store_type']
w = (X_val["perish"] * 0.25 + 1) / (X_val["perish"] * 0.25 + 1).mean()

for i in range(16):

    print("Step %d" % (i+1))

    dtrain = lgb.Dataset(
        X_train, label=y_train[:, i],
        categorical_feature=cate_vars,
        weight=None)
    dval = lgb.Dataset(
        X_val, label=y_val[:, i], reference=dtrain,
        weight=w,
        categorical_feature=cate_vars)
    bst = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dtrain, dval], verbose_eval=100)

    print("\n".join(("%s: %.2f" % x) for x in sorted(
        zip(X_train.columns, bst.feature_importance("gain")),
        key=lambda x: x[1], reverse=True)[:15]))
    best_rounds.append(bst.best_iteration or MAX_ROUNDS)

    val_pred.append(bst.predict(X_val, num_iteration=bst.best_iteration or MAX_ROUNDS))
    test_pred.append(bst.predict(X_test, num_iteration=bst.best_iteration or MAX_ROUNDS))
    gc.collect();

cal_score(y_val, np.array(val_pred).T)

make_submission(df_2017, np.array(test_pred).T)