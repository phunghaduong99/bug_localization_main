from sklearn.preprocessing import MinMaxScaler
import numpy as np
data = [[-1, 2, 3], [-0.5, 6, 3], [0, 10, 3], [1, 2, 3]]
print(data)

scaler = MinMaxScaler()
data2 = np.array(data).T

scaler.fit(data2)
print(scaler.transform(data2).T)


