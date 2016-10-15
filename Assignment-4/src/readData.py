from skimage import io 
import os
import numpy as np

#image = io.imread()
def load():
	dir = "../data/valid/"
	label = []
	with open(dir+'labels.txt','rb') as f:
		for line in f:
			label.append(int(line.split()[0]))
	l =  len(label)

	training_inputs = [np.reshape(io.imread(dir+str(i)+'.png'),(102400,1)) for i in range(l)]
	training_results = [vectorized_result(y) for y in label]
	# training_data = zip(np.asarray(training_inputs), np.asarray(training_results))
	return np.asarray(training_inputs), np.asarray(training_results)
	

#for file in os.listdir(dir):
 #   if file.endswith(".png"):
#	print file[:-4]
  #      image = io.imread(dir+file)
	#print image
	#io.imshow(image)
	
#io.show()
# load()

def vectorized_result(j):
	e = np.zeros((104, 1))
	e[j] = 1.0
	return e