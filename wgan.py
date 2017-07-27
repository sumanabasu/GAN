import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as pp
from PIL import Image

def init_weight(shape):
	nin = shape[-2]
	nout = shape[-1]
	if len(shape) == 2:
		k = np.sqrt(6./(nin + nout))
	elif len(shape) == 4:
		k = np.sqrt(6./(shape[2]*np.prod(shape[:2]) + shape[1]*np.prod(shape[:2])))
	initial = np.float32(np.random.uniform(-k,k, shape))
	return tf.Variable(initial)

def init_bias(shape):
  initial = tf.constant(0.0, shape=shape)
  return tf.Variable(initial)

'''
d_chnl_dim = [1, 256, 512, 1024]
g_chnl_dim = [1024, 512, 256, 1]
z_dim = 100
'''
d_chnl_dim = [1, 16, 16, 64]
g_chnl_dim = [64, 16, 16, 1]
z_dim = 64
filter_size = 3

d_weights = {
'w1' : init_weight([filter_size, filter_size, d_chnl_dim[0], d_chnl_dim[1]]),
'w2' : init_weight([filter_size, filter_size, d_chnl_dim[1], d_chnl_dim[2]]),
'w3' : init_weight([filter_size, filter_size, d_chnl_dim[2], d_chnl_dim[3]]), 
'w4' : init_weight([4 *  4 * d_chnl_dim[3], 1]),

'b1' : init_bias([d_chnl_dim[1]]),
'b2' : init_bias([d_chnl_dim[2]]),
'b3' : init_bias([d_chnl_dim[3]]), 
'b4' : init_bias([d_chnl_dim[3]])
}

g_weights = {
'w1' : init_weight([z_dim, g_chnl_dim[0] * 4 * 4]),
'w2' : init_weight([filter_size, filter_size, g_chnl_dim[1], g_chnl_dim[0]]),
'w3' : init_weight([filter_size, filter_size, g_chnl_dim[2], g_chnl_dim[1]]), 
'w4' : init_weight([filter_size, filter_size, g_chnl_dim[3], g_chnl_dim[2]]),

'b1' : init_bias([g_chnl_dim[0] * 4 * 4]),
'b2' : init_bias([g_chnl_dim[1]]),
'b3' : init_bias([g_chnl_dim[2]]),
'b4' : init_bias([g_chnl_dim[3]])
}


'''
def batch_norm(x, n_out, phase_train):
	with tf.variable_scope('bn'):
		beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta', trainable=True)
		gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma', trainable=True)
		batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
		ema = tf.train.ExponentialMovingAverage(decay=0.5)

		def mean_var_with_update():
			ema_apply_op = ema.apply([batch_mean, batch_var])
			with tf.control_dependencies([ema_apply_op]):
				return tf.identity(batch_mean), tf.identity(batch_var)

		mean, var = tf.cond(phase_train,mean_var_with_update,  lambda: (ema.average(batch_mean), ema.average(batch_var)))
		normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
	return normed
'''

def leaky_relu(x, alpha=0.1, name="LeakyReLU"):
	return tf.maximum(x, alpha * x)

def gen_z(batch_size, z_dim):
	#return tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name='z')
	return np.random.normal(size=(batch_size, z_dim))

def D(x, d_w = d_weights):

	w1 = d_w['w1']
	w2 = d_w['w2']
	w3 = d_w['w3']
	w4 = d_w['w4']

	b1 = d_w['b1']
	b2 = d_w['b2']
	b3 = d_w['b3']
	b4 = d_w['b4']

	chnl = 32
	#chnl_dim = [1, 256, 512, 1024]
	global d_chnl_dim
	chnl_dim = d_chnl_dim

	x_image = tf.reshape(x, [-1, 32, 32, 1])
	#w1 = init_weight([5, 5, chnl_dim[0], chnl_dim[1]])
	#b1 = init_bias([chnl_dim[1]])
	h1 = tf.nn.conv2d(input = x_image, filter = w1, strides=[1, 2, 2, 1], padding='SAME') + b1
	#h1 = tf.nn.relu(h1)
	h1 = leaky_relu(h1)
	#h1 = batch_norm(h1, chnl_dim[1] , tf.Variable(True))
	h1 = tf.contrib.layers.batch_norm(h1, epsilon = 1e-5)

	#w2 = init_weight([5, 5, chnl_dim[1], chnl_dim[2]])
	#b2 = init_bias([chnl_dim[2]])
	h2 = tf.nn.conv2d(input = h1, filter = w2, strides=[1, 2, 2, 1], padding='SAME') + b2
	#h2 = tf.nn.relu(h2)
	h2 = leaky_relu(h2)
	#h2 = batch_norm(h2, chnl_dim[2], tf.Variable(True))
	h2 = tf.contrib.layers.batch_norm(h2, epsilon = 1e-5)

	#w3 = init_weight([5, 5, chnl_dim[2], chnl_dim[3]])
	#b3 = init_bias([chnl_dim[3]])
	h3 = tf.nn.conv2d(input = h2, filter = w3, strides=[1, 2, 2, 1], padding='SAME') + b3
	#h3 = tf.nn.relu(h3)
	h3 = leaky_relu(h3)
	#h3 = batch_norm(h3, chnl_dim[3], tf.Variable(True))
	h3 = tf.contrib.layers.batch_norm(h3, epsilon = 1e-5)

	#w4 = init_weight([4 *  4 * chnl_dim[3], 1])
	#b4 = init_bias([4 *  4 * chnl_dim[3]])
	h3 = tf.reshape(h3, [-1, 4 *  4 * chnl_dim[3]])
	h4 = tf.matmul(h3, w4) + b4
	#h4 = tf.nn.sigmoid(h4)

	return h4

def G(z, g_w = g_weights):

	w1 = g_w['w1']
	w2 = g_w['w2']
	w3 = g_w['w3']
	w4 = g_w['w4']

	b1 = g_w['b1']
	b2 = g_w['b2']
	b3 = g_w['b3']
	b4 = g_w['b4']

	#print w1.get_shape(), w2.get_shape(), w3.get_shape(), w4.get_shape()
	#print b1.get_shape(), b2.get_shape(), b3.get_shape(), b4.get_shape()

	chnl = 32
	batch_size, z_dim = tf.shape(z)[0], tf.shape(z)[1]
	#chnl_dim = [1024, 512, 256, 1]
	global g_chnl_dim
	chnl_dim = g_chnl_dim

	#1024 x 4 x 4
	#w1 = init_weight([z_dim, chnl_dim[0] * 4 * 4])
	#w1 =m tf.reshape(w1, [-1, 4, 4, 1024])
	h1 = tf.matmul(z, w1) + b1
	#h1 = tf.contrib.layers.batch_norm(h1, epsilon = 1e-5)
	h1 = tf.nn.relu(h1)
	h1 = tf.reshape(h1, [batch_size, 4, 4, chnl_dim[0]])

	#512 x 8 x 8
	#w2 = init_weight([5, 5, chnl_dim[1], chnl_dim[0]])
	out = [batch_size, 8, 8, chnl_dim[1]]
	h2 = tf.nn.conv2d_transpose(value = h1, filter = w2, output_shape = out, strides = [1, 2, 2, 1], padding = 'SAME')
	h2 = tf.reshape(h2, out) + b2
	h2 = tf.nn.relu(h2)#tf.contrib.layers.batch_norm(h2, epsilon = 1e-5))

	#128 x 16 x 16
	#w3 = init_weight([5, 5, chnl_dim[2], chnl_dim[1]])
	out = [batch_size, 16, 16, chnl_dim[2]]
	h3 = tf.nn.conv2d_transpose(value = h2, filter = w3, output_shape = out, strides = [1, 2, 2, 1], padding = 'SAME')
	h3 = tf.reshape(h3, out) + b3
	h3 = tf.nn.relu(h3) #tf.contrib.layers.batch_norm(h3, epsilon = 1e-5))

	#1 x 32 x 32
	#w4 = init_weight([5, 5, chnl_dim[3], chnl_dim[2]])
	out = [batch_size, 32, 32, chnl_dim[3]]
	h4 = tf.nn.conv2d_transpose(value = h3, filter = w4, output_shape = out, strides = [1, 2, 2, 1], padding = 'SAME')
	h4 = tf.reshape(h4, out) + b4
	h4 = tf.nn.sigmoid(h4)

	return h4

	#z = tf.truncated_normal([batch_size, z_dim], mean = 0, stddev = 1, name = 'z')
	#g_w1 = tf.get_variable('g_w1', [z_dim, nch * 4 * 4], dtype = tf.float32, initializer = tf.truncated_normal_initializer(stddev = 0.02))

def train(images):

	# Hyper-parameters
	epochs = 1000
	batch_size = 256
	z_dim = 64
	x_size = 32 * 32
	regularization = 1.2 

	#z = gen_z(batch_size, z_dim)
	z = tf.placeholder('float', shape=(None, z_dim))
	X = tf.placeholder('float', shape=(None, x_size))

	sample = G(z)
	DGZ = D(sample)
	DX = D(X)

	input_gradient_gz = tf.reduce_mean(tf.gradients(tf.reduce_mean(DGZ),
													sample)[0]**2)
	#G_objective = -tf.reduce_mean(tf.log(DGZ))
	G_objective = -tf.reduce_mean(DGZ) + regularization * input_gradient_gz
	#D_objective = -tf.reduce_mean(tf.log(D(X)) + tf.log(1 - D(G(z))))

	input_gradient_x = tf.reduce_mean(tf.gradients(tf.reduce_mean(DX),
													X)[0]**2)
	D_objective = -tf.reduce_mean(DX - DGZ) + regularization * input_gradient_x

	#print (G_objective.get_shape().as_list(), D_objective.get_shape().as_list())

	G_opt = tf.train.AdamOptimizer(learning_rate = 0.00002).minimize(G_objective, var_list=g_weights.values())
	D_opt = tf.train.AdamOptimizer(learning_rate = 0.000005).minimize(D_objective, var_list=d_weights.values())


	G_losses = []
	D_losses = []

	# Session
	config = tf.ConfigProto(log_device_placement=False)
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(epochs):
			#print "Epoch haha : ", epoch

			batch = 0

			#real image batch
			miniBatch = createMinibatches(images, batch_size)
			imgBatch = miniBatch()
			

			for realImgBatch in imgBatch:
				print "Epoch : ", epoch, ", Batch : ", batch
				batch += 1

				g = sess.run([G_opt, G_objective], feed_dict={
					z: gen_z(batch_size, z_dim)
					})

				G_losses.append(g[1])

				for i in range(1):
					d = sess.run([D_opt, D_objective, 
							#tf.reduce_mean(DGZ), 
							#tf.reduce_mean(DX)
							],
							feed_dict={
								X: realImgBatch.reshape(-1, x_size),
								z: gen_z(realImgBatch.shape[0], z_dim)
								})
					D_losses.append(d[1])

			#print 'epoch', epoch, epoch % 100 == 0
			# Show a random image
			if epoch % 5 == 0:
				fake_image = sess.run(sample, feed_dict={
					z:gen_z(8, z_dim)
					})
				pp.clf()
				pp.plot(G_losses, label='G losses')
				pp.plot(D_losses, label='D losses')
				pp.legend()
				pp.savefig('losses.png')
				fake_image *= 255.
				#print fake_image.shape
				fake_image = np.hstack(fake_image.reshape(8, 32, 32))
				Image.fromarray(fake_image.astype(np.uint8)).save(str(epoch)+"_"+".png")


def read_image():
	mnist = input_data.read_data_sets("MNIST_data/")
	images = mnist.train.images
	#print(images.min(), images.max(), images.mean())
	images = np.reshape(images, (images.shape[0], 28, 28))
	#print images.shape
	img = np.zeros((images.shape[0], 32, 32),'float32')
	img[:, :28, :28] = images.reshape(images.shape[0], 28, 28)
	#print img.shape
	#print(img.min(), img.max(), img.mean())
	return img

def createMinibatches(img, mbSize):
	np.random.shuffle(img)

	idx = np.arange(0, img.shape[0], mbSize)

	def genMinibatch():
		for i in idx:
			yield img[i : i + mbSize]

	return genMinibatch
 
def main():
	images = read_image()
	train(images)


if __name__ == "__main__":
	main()
