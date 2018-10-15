import numpy as np
import pandas as pd
from scipy import stats

x1 = pd.read_csv('vote/labelled.csv') # 0.920
x2 = pd.read_csv('vote/labelled1.csv') # 0.918
x3 = pd.read_csv('vote/labelled2.csv') # 0.918
x4 = pd.read_csv('vote/labelled3.csv') # 0.9265
x5 = pd.read_csv('vote/labelled4.csv') # 0.93
x6 = pd.read_csv('vote/labelled5.csv') # 0.95
x7 = pd.read_csv('vote/labelled6.csv') # 0.95
x8 = pd.read_csv('vote/labelled7.csv') # 0.95
x9 = pd.read_csv('vote/labelled8.csv') # 0.95
x10 = pd.read_csv('vote/labelled9.csv') # 0.95
x11 = pd.read_csv('vote/labelled10.csv') # 0.95
x12 = pd.read_csv('vote/labelled11.csv') # 0.95
x13 = pd.read_csv('vote/labelled12.csv') # 0.95
x14 = pd.read_csv('vote/labelled13.csv') # 0.95
x15 = pd.read_csv('vote/labelled14.csv') # 0.95
x16 = pd.read_csv('vote/labelled15.csv') # 0.95
x17 = pd.read_csv('vote/labelled16.csv') # 0.95
x18 = pd.read_csv('vote/labelled17.csv') # 0.95
x19 = pd.read_csv('vote/labelled18.csv') # 0.95
#---###较

y1 = np.array(x1['y']).reshape(8000,1)
y2 = np.array(x2['y']).reshape(8000,1)
y3 = np.array(x3['y']).reshape(8000,1)
y4 = np.array(x4['y']).reshape(8000,1)
y5 = np.array(x5['y']).reshape(8000,1)
y6 = np.array(x6['y']).reshape(8000,1)
y7 = np.array(x7['y']).reshape(8000,1)
y8 = np.array(x8['y']).reshape(8000,1)
y9 = np.array(x9['y']).reshape(8000,1)
y10 = np.array(x10['y']).reshape(8000,1)
y11 = np.array(x11['y']).reshape(8000,1)
y12 = np.array(x12['y']).reshape(8000,1)
y13 = np.array(x13['y']).reshape(8000,1)
y14 = np.array(x14['y']).reshape(8000,1)
y15 = np.array(x15['y']).reshape(8000,1)
y16 = np.array(x16['y']).reshape(8000,1)
y17 = np.array(x17['y']).reshape(8000,1)
y18 = np.array(x18['y']).reshape(8000,1)
y19 = np.array(x19['y']).reshape(8000,1)

#
y = np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12,y13,y14,y15,y16,y17,y18,y19),axis=1)
# print(y.shape)
#
y_test = np.array(stats.mode(y,axis=1)[0]).reshape(8000,)
print(y_test.shape)
result = pd.DataFrame(columns=['Id','y'])
result['y'] = y_test
result['Id'] = range(30000,38000)
result.to_csv('vote3.csv',index=False,header=True)