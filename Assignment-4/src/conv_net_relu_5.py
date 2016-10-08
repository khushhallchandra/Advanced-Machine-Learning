import tensorflow as tf
import numpy as np
import input_data
import smallInputData

batch_size = 128
test_size = 1828#256

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def batch_norm(x, n_out, phase_train, scope='bn'):
    with tf.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed


def model(X, w, w2, w3, w4, w5, w_o, p_keep_conv, p_keep_hidden,im_size,phase_train):

    l1a = tf.nn.relu(batch_norm(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'),im_size,phase_train)) # l1a shape=(?, 28, 28, 32)
                        
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # l1 shape=(?, 14, 14, 32)
                        
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(batch_norm(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'),im_size*2,phase_train)) # l2a shape=(?, 14, 14, 64)
                        
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # l2 shape=(?, 7, 7, 64)
                        
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(batch_norm(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'),im_size*4,phase_train)) # l2a shape=(?, 14, 14, 64)
                        
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') # l2 shape=(?, 7, 7, 64)
                        
    l3 = tf.nn.dropout(l3, p_keep_conv)

   
    l4a = tf.nn.relu(batch_norm(tf.nn.conv2d(l3, w4,strides=[1, 1, 1, 1], padding='SAME'),im_size*8,phase_train)) # l3a shape=(?, 7, 7, 128)
                        
    l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') # l3 shape=(?, 4, 4, 128)

    print(w5.get_shape().as_list(),"----------------------------------")                        
    l4 = tf.reshape(l4, [-1, w5.get_shape().as_list()[0]])    # reshape to (?, 2048)
    l4 = tf.nn.dropout(l4, p_keep_conv)

    l5 = tf.nn.relu(tf.matmul(l4, w5))
    l5 = tf.nn.dropout(l5, p_keep_hidden)

    pyx = tf.matmul(l5, w_o)
    return pyx

mnist = input_data.read_data_sets("../data/train/","../data/valid/")
#mnist = smallInputData.read_data_sets("dummy/","dummy1/")
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 32, 32, 1)  # 28x28x1 input img
teX = teX.reshape(-1, 32, 32, 1)  # 28x28x1 input img

X = tf.placeholder("float", [None, 32, 32, 1])
Y = tf.placeholder("float", [None, 104])
phase_train = tf.placeholder(tf.bool, name='phase_train')

#w = init_weights([8, 8, 1, 32])       # 3x3x1 conv, 32 outputs
#w2 = init_weights([5, 5, 32, 64])     # 3x3x32 conv, 64 outputs
#w3 = init_weights([3, 3, 64, 128])    # 3x3x32 conv, 128 outputs
#w4 = init_weights([128 *4*4, 1000]) # FC 128 * 4 * 4 inputs, 625 outputs
#w_o = init_weights([1000, 104])         # FC 625 inputs, 104 outputs (labels)

#w = init_weights([8, 8, 1, 32])       # 3x3x1 conv, 32 outputs
#w2 = init_weights([5, 5, 32, 64])     # 3x3x32 conv, 64 outputs
w = init_weights([6, 6, 1, 32])       # 3x3x1 conv, 32 outputs
w2 = init_weights([4, 4, 32, 64])     # 3x3x32 conv, 64 outputs
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([3, 3, 128, 256])    # 3x3x32 conv, 128 outputs
w5 = init_weights([256 *2*2, 1000]) # FC 128 * 4 * 4 inputs, 625 outputs
w_o = init_weights([1000, 104])         # FC 625 inputs, 104 outputs (labels)


p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, w3, w4, w5, w_o, p_keep_conv, p_keep_hidden,32,phase_train)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.003, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver() 

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(60):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5,phase_train: True})
	#print 
#    saver.restore(sess, "model/model.ckpt")
#    print "restored"
	test_indices = np.arange(len(teX)) # Get A Test Batch
	np.random.shuffle(test_indices)
	test_indices = test_indices[0:test_size]

    	print (i, np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX[test_indices], Y: teY[test_indices],
                                                         	p_keep_conv: 1.0, p_keep_hidden: 1.0,phase_train: True}))*100.0)
    #save_path = saver.save(sess, "model/model.ckpt")
    #print("Model saved in file: %s" % save_path)

