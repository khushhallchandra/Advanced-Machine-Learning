from skimage import io 
import os
import numpy
from scipy import misc

num_classes=104
def extract_images(dir,N):
    training_inputs = numpy.asarray([misc.imresize((255.0 - io.imread(dir+str(i)+'.png'))/255.0,(28,28)) for i in range(N)])
    return training_inputs.reshape(N, 784)

def extract_labels(dir):
    labels = []
    with open(dir+'labels.txt','rb') as f:
        for line in f:
            labels.append(int(line.split()[0]))
    labels_dense = numpy.asarray(labels,dtype=numpy.uint8)
    num_labels = labels_dense.shape[0]
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    for i in range(num_labels):
	labels_one_hot[i][(labels_dense[i])] =1
    return labels_one_hot

def images(tr_dir, va_dir):

    train_labels = extract_labels(tr_dir)
    N = train_labels.shape[0]
    train_images = extract_images(tr_dir,N)
    
    test_labels = extract_labels(va_dir)
    N = test_labels.shape[0]
    test_images = extract_images(va_dir,N)

    print "Data reading complete"

    return train_images,train_labels,test_images,test_labels
