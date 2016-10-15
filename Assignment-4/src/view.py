import smallInputData
import input_data
import scipy.misc as ms
import numpy as np

#a = smallInputData.read_data_sets('dummy/','dummy1/')
a = input_data.read_data_sets('../data/train/','../data/valid/')
print "Press -1 to exit any time"

f = input("Enter image no. = ")
while(f!=-1):
	#ms.imshow(np.reshape(a.train.images[f],(28,28)))
        ms.imshow(np.reshape(a.train.images[f],(32,32)))
	f = input("Enter image no. = ")
