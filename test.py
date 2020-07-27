import numpy as np 
a=np.array([[[1,2,3]]])
b=np.array([[[5,6,7]]])
print (np.concatenate((a,b),axis=1 ).shape)

c=np.array([[1,2,3],[3,4,5]])
print(c[0,:])
print(c[0])
d=np.array([[[]]])

print(d.shape)