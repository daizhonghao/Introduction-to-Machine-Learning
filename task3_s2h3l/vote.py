import numpy as np
import pandas as pd
from scipy import stats

x1 = pd.read_csv('tf_6.csv') # 0.920
x2 = pd.read_csv('tf_7.csv') # 0.918
x3 = pd.read_csv('tf_8.csv') # 0.918
x4 = pd.read_csv('tf_9.csv') # 0.9265
x5 = pd.read_csv('tf_10.csv') # 0.93
x6 = pd.read_csv('tf_11.csv') # 0.95
x7 = pd.read_csv('tf_12.csv') # 0.98
x8 = pd.read_csv('tf_13.csv') # 0.98
#---###较

x9 = pd.read_csv('tf_4.csv') # 0.910
x10 = pd.read_csv('tf_5.csv') # 0.915
x11 = pd.read_csv('MPL_1_layer.csv')
x12 = pd.read_csv('MPL_4_layer.csv')
x13 = pd.read_csv('MPL_less_deep.csv') # .90
x14 = pd.read_csv('tensor.csv') # 0.85

y1 = np.array(x1['y']).reshape(8137,1)
y2 = np.array(x2['y']).reshape(8137,1)
y3 = np.array(x3['y']).reshape(8137,1)
y4 = np.array(x4['y']).reshape(8137,1)
y5 = np.array(x5['y']).reshape(8137,1)
y6 = np.array(x6['y']).reshape(8137,1)
y7 = np.array(x7['y']).reshape(8137,1)
y8 = np.array(x8['y']).reshape(8137,1)
y9 = np.array(x9['y']).reshape(8137,1)
y10 = np.array(x10['y']).reshape(8137,1)
y11 = np.array(x11['y']).reshape(8137,1)
y12 = np.array(x12['y']).reshape(8137,1)
y13 = np.array(x13['y']).reshape(8137,1)
y14 = np.array(x14['y']).reshape(8137,1)
y = np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14),axis=1)
print(y.shape)

y_test = np.array(stats.mode(y,axis=1)[0]).reshape(8137,)
print(y_test.shape)
result = pd.DataFrame(columns=['Id','y'])
result['y'] = y_test
result['Id'] = range(45324,53461)
result.to_csv('vote.csv',index=False,header=True)

