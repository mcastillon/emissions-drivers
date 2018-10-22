import lightgbm as lgb
import matplotlib.pyplot as plt
import os
import pandas as pd
from urllib2 import urlopen


states = []

nums = range(1, 57)
exclude = [3, 7, 14, 43, 52]
ids = [n for n in nums if n not in exclude]


for i in ids:
    p = str(i).zfill(2)
    url = urlopen('https://htaindex.cnt.org/download/download.php?focus=blkgrp&geoid={0}'.format(p))
    output = open('zipFile.zip', 'wb')
    output.write(url.read())
    output.close()
    data = pd.read_csv('zipFile.zip')
    states.append(data)
    os.remove('zipFile.zip')

all_states = pd.concat(states)
all_states = all_states[~pd.isnull(all_states['co2_per_hh_local'])]
all_states['co2_per_capita'] = all_states['co2_per_hh_local']*all_states['households']/all_states['population']

params = {
    'num_threads': 8,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'num_leaves': 100,
    'max_depth': 66,
    'max_drop': -1,
    'learning_rate': 0.01,
    'feature_fraction': 0.333,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'tree_learner': 'feature',
    'lambda_l1': 1,
    'verbose': 0,
    'max_bin': 128
}

data = lgb.Dataset(all_states.drop(['co2_per_hh_local',
                                    'co2_per_acre_local',
                                    'blkgrp',
                                    'blkgrps',
                                    'cbsa',
                                    'co2_per_capita'], axis=1),
                   all_states['co2_per_capita'])

gbm = lgb.train(params,
                data,
                num_boost_round=1000)

lgb.plot_importance(gbm)
plt.show()