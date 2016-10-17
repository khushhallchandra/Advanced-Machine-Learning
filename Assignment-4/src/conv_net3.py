import tensorflow as tf
import numpy as np
import try1

batch_size = 128
test_size = 1828

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


def model(X, w, w2, wn, w3, w4, w_o, p_keep_conv, p_keep_hidden,im_size,phase_train):

    l1a = tf.nn.relu6(batch_norm(tf.nn.conv2d(X, w, strides=[1, 1, 1, 1], padding='SAME'),im_size,phase_train))
                        
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu6(batch_norm(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'),im_size*2,phase_train)) 
                        
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME') 
                        
    l2 = tf.nn.dropout(l2, p_keep_conv)

    lna = tf.nn.relu6(batch_norm(tf.nn.conv2d(l2, wn, strides=[1, 1, 1, 1], padding='SAME'),im_size*4,phase_train))
                        
    ln = tf.nn.max_pool(lna, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
                        
    ln = tf.nn.dropout(ln, p_keep_conv)

   
    l3a = tf.nn.relu6(batch_norm(tf.nn.conv2d(ln, w3,strides=[1, 1, 1, 1], padding='SAME'),im_size*8,phase_train)) 
                        
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME') 
                      
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])  
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)

    pyx = tf.matmul(l4, w_o)
    return pyx

trX, trY, teX, teY = try1.images("../data/train/","../data/valid/")
imsize = 28
trX = trX.reshape(-1, imsize, imsize, 1)
teX = teX.reshape(-1, imsize, imsize, 1)

X = tf.placeholder("float", [None, imsize, imsize, 1])
Y = tf.placeholder("float", [None, 104])
phase_train = tf.placeholder(tf.bool, name='phase_train')

w = init_weights([6, 6, 1, 32])       
w2 = init_weights([4, 4, 32, 64])     
wn = init_weights([3, 3, 64, 128])
w3 = init_weights([3, 3, 128, 256])   
w4 = init_weights([256 *2*2, 1000]) 
w_o = init_weights([1000, 104])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w, w2, wn, w3, w4, w_o, p_keep_conv, p_keep_hidden,32,phase_train)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.RMSPropOptimizer(0.003, 0.9).minimize(cost)
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver() 
out = np.zeros(100)
# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for i in range(35):
        training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end], p_keep_conv: 0.8, p_keep_hidden: 0.5,phase_train: True})

	test_indices = np.arange(len(teX))
	np.random.shuffle(test_indices)
	test_indices = test_indices[0:test_size]
        
	out[i] = np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices], Y: teY[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0,phase_train: True}))*100.0

    	print (i, out[i])
	if(out[i]>94.8):
	    break
    save_path = saver.save(sess, "model/model.ckpt")
    print("Model saved in file: %s" % save_path)
    print out
