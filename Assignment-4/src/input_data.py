"""Functions for downloading and reading MNIST data."""
from skimage import io 
import os
import numpy

def extract_images(dir,N):
    # dir = "../data/valid/"
    # N
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    training_inputs = numpy.asarray([io.imread(dir+str(i)+'.png') for i in range(N)])
    (x,y,z) = training_inputs.shape
    training_inputs = training_inputs.reshape(x, y, z, 1)
    return training_inputs


def dense_to_one_hot(labels_dense, num_classes=104):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


# def extract_labels(filename, one_hot=False):
def extract_labels(dir):
    labels = []
    with open(dir+'labels.txt','rb') as f:
        for line in f:
            labels.append(int(line.split()[0]))
    labels = numpy.asarray(labels,dtype=numpy.uint8)
    return dense_to_one_hot(labels)



class DataSet(object):
    def __init__(self, images, labels):

        assert images.shape[0] == labels.shape[0], (
            "images.shape: %s labels.shape: %s" % (images.shape,
                                                   labels.shape))
        self._num_examples = images.shape[0]
        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, rows*columns] (assuming depth == 1)
        assert images.shape[3] == 1
        images = images.reshape(images.shape[0],
                                images.shape[1] * images.shape[2])
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir):
    class DataSets(object):
        pass
    data_sets = DataSets()

    dir = "../data/valid/"

    train_labels = extract_labels(train_dir)
    N = train_labels.shape[0]
    train_images = extract_images(train_dir,N)

    data_sets.train = DataSet(train_images, train_labels)
    # data_sets.validation = DataSet(validation_images, validation_labels)
    # data_sets.test = DataSet(test_images, test_labels)
    return data_sets