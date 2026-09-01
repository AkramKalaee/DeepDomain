import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

from constants import SRC_DIR

results = {}
model = 'DAVE2'
with open(f'{SRC_DIR}/tools/DEEPDOMAIN/AI_self_driving_car_main/predictions/predictions_{model}.json') as f:
  data_preds = json.load(f)

df = pd.read_csv(f'{SRC_DIR}/dataset/test/labels.csv', usecols=["steering_angle"])
labels = np.array(list(df.values))

df.tail()

preds = np.array(list(data_preds.values()))

mse = mean_squared_error(y_true=labels, y_pred=preds)
print(mse)
exit(0)



from statsmodels.tsa.api import SimpleExpSmoothing, Holt
fit1 = SimpleExpSmoothing(preds).fit(smoothing_level=0.4,optimized=True)

preds = fit1.fittedvalues

preds.max()


preds_public = preds[df['public'] == 1].reshape(-1)

#%%

scores_public = df[df['public'] == 1].steering_angle.values

#%%

results['public'] = np.sqrt(((preds_public - scores_public) ** 2).mean())

#%%

import matplotlib.pyplot as plt
plt.plot(range(len(preds_public)),preds_public, c='r')
plt.plot(range(len(preds_public)),scores_public, c='g')
plt.title(f'{model} public test data')
plt.legend(["predictions", "ground truth"], loc ="lower right")
plt.ylabel('steering angle')
plt.savefig(f'{model}_public.png')

#%%

preds_private = preds[df['public'] == 0].reshape(-1)

#%%

scores_private = df[df['public'] == 0].steering_angle.values

#%%

results['private'] = np.sqrt(((scores_private - preds_private) ** 2).mean())

#%%

import matplotlib.pyplot as plt
plt.plot(range(len(preds_private)),preds_private, c='r' )
plt.plot(range(len(scores_private)),scores_private, c='g' )
plt.title(f'{model} pivate test data')
plt.legend(["predictions", "ground truth"], loc ="lower right")
plt.ylabel('steering angle')
plt.savefig(f'{model}_private.png')

#%% md



#%% md

# Smoothed Results Transformer + Optical Flow

#%%

results

#%% md

# True results Transformer + Optical Flow



#%%

results

#%% md

# Simple Transformer

#%%

results

#%% md

# Transfer Learning

#%%

results

#%% md

# DAVE2

#%%

results

#%% md

# CNN-LSTM

#%%

results

#%%

import matplotlib.pyplot as plt

plt.plot(range(len(train_data)),train_data, c='g' )
plt.title(f'Training Data')
plt.legend([ "ground truth"], loc ="lower right")
plt.ylabel('steering angle')
plt.savefig('train_data.png')
