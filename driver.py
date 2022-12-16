import joblib
import numpy as np


#read the sample file for which the classifications are needed
f = open('SampleTest.csv','r')
data = f.read()
f.close()

data = data.split('\n')

X_test = []

for i in range(0,len(data)):
    line = data[i].strip().split(',')
    if(len(line[0]) != 0):
        for j in range(0, len(line)):
            line[j] = float(line[j])
        X_test.append(line)
        
X_test = np.array(X_test)

#load the scaler from the training set
scaler = joblib.load('scaler.pkl')
# apply same scale transformation to test data
X_test = scaler.transform(X_test)

#load the model
clf = joblib.load('svmpoly.pkl')
#perform the prediction
print("[CONFUSING, CLEAR]")
print(clf.predict_proba(X_test))
