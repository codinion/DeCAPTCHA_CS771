""" 
This script is responsible for training a machine learning model. 
Here a RandomForestClassifier is being trained.
Other models may be used interchangeably too.
Author : Abir Mukherjee
Email : abir.mukherjee0595@gmail.com
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle 

data=np.genfromtxt('data.out',delimiter=',', dtype= np.float32)
X=data[:,:-1]
Y=np.int32(data[:,-1])
clf=RandomForestClassifier(n_estimators=26)
clf.fit(X,Y)

filename = 'model'
pickle.dump(clf, open(filename, 'wb'))