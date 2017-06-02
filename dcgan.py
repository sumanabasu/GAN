
from keras.layers import Input, Dense, Activation, Reshape, Dropout, Flatten
from keras.layers.normalization import *
from keras.layers.convolutional import UpSampling2D, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam, SGD
import struct
import numpy as np
from keras.layers.advanced_activations import LeakyReLU
import matplotlib.pyplot as plt
from keras.utils import vis_utils
from keras.models import Sequential
from PIL import Image
import math
import os
import cPickle

#os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu,floatX=float32"

img = np.empty(shape = (0,0))
loss = {"d" : [], "g" : []}
mean_ = 0
std_ = 0

def generator():
	nch = 512	#number of channels
	z_dim = 100
	g_input = Input(shape = [z_dim])

	#1024 x 4 x 4
	H = Dense(units = nch*4*4, kernel_initializer = 'glorot_normal')(g_input)
	H = BatchNormalization(momentum = 0.9, epsilon = 0.01)(H)
	H = Activation('relu')(H)
	H = Reshape([nch, 4, 4])(H)

	#512 x 8 x 8
	#H = UpSampling2D(size = (2,2), data_format = 'channels_first')(H)
	H = Conv2DTranspose(filters = nch/2, kernel_size = (5,5), data_format = 'channels_first',\
		strides = (2,2), padding = 'same', kernel_initializer = 'glorot_uniform')(H)
	H = BatchNormalization(momentum = 0.9, epsilon = 0.01)(H)
	H = Activation('relu')(H)

	# 256 x 16 x 16
	#H = UpSampling2D(size = (2,2), data_format = 'channels_first')(H)
	H = Conv2DTranspose(filters = nch/4, kernel_size = (5,5), data_format = 'channels_first',\
		strides = (2,2), padding = 'same', kernel_initializer = 'glorot_uniform')(H)
	H = BatchNormalization(momentum = 0.9, epsilon = 0.01)(H)
	H = Activation('relu')(H)

	#128 x 32 x 32
	#H = UpSampling2D(size = (2,2), data_format = 'channels_first')(H)
	H = Conv2DTranspose(filters = 1, kernel_size = (5,5), data_format = 'channels_first',\
		strides = (2,2), padding = 'same', kernel_initializer = 'glorot_uniform')(H)
	#H = BatchNormalization(momentum = 0.9, epsilon = 0.01)(H)
	g_V = Activation('tanh')(H)

	generator = Model(g_input, g_V)
	#generator.compile(loss = 'binary_crossentropy', optimizer = Adam(lr = 0.0001, decay = 0.0))
	generator.summary()
	return generator

def discriminator():

	d_input = Input(shape = (1,32,32))
	nch = 128

	#128 x 16 x 16
	H = Conv2D(filters = nch, kernel_size = (5,5), strides = (2,2), padding = 'same', \
		data_format = 'channels_first')(d_input)
	H = BatchNormalization(momentum = 0.9)(H)
	H = LeakyReLU(0.2)(H)
	#H = Dropout(0.2)(H)

	#256 x 8 x 8
	H = Conv2D(filters = nch*2, kernel_size = (5,5), strides = (2,2), padding = 'same', \
		data_format = 'channels_first')(H)
	H = BatchNormalization(momentum = 0.9)(H)
	H = LeakyReLU(0.2)(H)
	#H = Dropout(0.2)(H)

	#512 x 4 x 4
	H = Conv2D(filters = nch*4, kernel_size = (5,5), strides = (2,2), padding = 'same', \
		data_format = 'channels_first')(H)
	#H = BatchNormalization(momentum = 0.9)
	#H = LeakyReLU(0.2)(H)
	#H = Dropout(0.2)(H)

	H = Flatten()(H)
	d_V = Dense(1, activation = 'sigmoid')(H)
	
	discriminator = Model(d_input,d_V)
	#discriminator.compile(loss='categorical_crossentropy', optimizer=Adam(lr = 0.001, decay = 0.0))
	discriminator.summary()
	return discriminator

def GAN(gen, disc):
	#gen = generator()
	#disc = discriminator()

	#disc.trainable = False
	gan_input = Input(shape=[100])
	H = gen(gan_input)
	gan_V = disc(H)
	gan_V.trainable = False
	GAN = Model(gan_input, gan_V)
	#GAN.compile(loss = 'categorical_crossentropy', optimizer = Adam(lr = 0.001, decay = 0.0))
	GAN.summary()
	return GAN


def read_data():
	with open('../data/train-images.idx3-ubyte', 'rb') as fl:
		magic, num, rows, cols = struct.unpack(">IIII", fl.read(16))
		img_train = np.fromfile(fl, dtype = np.int8).reshape(num, rows, cols)
	#print img.shape

	img = np.zeros((img_train.shape[0], 1, 32, 32),'float32')
	img[:, :, :28, :28] = img_train.reshape(img_train.shape[0], 1, rows, cols)

	print img.shape


	#img = img.astype('float32') - np.min(img)/(np.max(img) - np.min(img))
	#img /= 255
	global min_, std_
	mean_ = np.mean(img)
	std_ = np.std(img)

	# img = (img.astype('float32') - np.mean(img)) / np.std(img)
	img /= 255. 
	img -= 0.5 
	img /= 0.5
	print np.min(img), np.max(img)

	return img

def createMinibatches(mbSize):
	global img
	np.random.shuffle(img)

	idx = np.arange(0, img.shape[0], mbSize)

	def genMinibatch():
		for i in idx:
			yield img[i : i + mbSize]

	return genMinibatch

def plot_loss():
	plt.figure()
	plt.plot(loss["d"], label = 'discriminative loss')
	plt.plot(loss["g"], label = 'generative loss')
	plt.savefig('loss.png')
	plt.legend()
	plt.show()

def plot_gen(n_ex=16,dim=(4,4), figsize=(10,10) ):
    noise = np.random.uniform(0,1,size=[n_ex,100])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,0,:,:]
        plt.imshow(img)
        plt.axis('off')
    plt.tight_layout()
    plt.show()


def plotGeneratedImage(fakeImgBatch):
	plt.figure(figsize = (10, 10))
	for i in range(fakeImgBatch.shape[0]):
		#plt.subplot()
		img = fakeImgBatch[i,0,:,:]
		plt.imshow(img)
	plt.show()

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[2:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[0, :, :]
    return image

def train(mbSize, n_epoch, nzLen):
	global loss
	d_optim = SGD(lr = 1e-3, momentum=0.9)
	g_optim = Adam(lr = 1e-4)

	gen = generator()
	disc = discriminator()
	gan_model = GAN(gen, disc)

	gen.compile(loss = 'binary_crossentropy', optimizer = g_optim)
	gan_model.compile(loss = 'binary_crossentropy', optimizer = g_optim)
	disc.trainable = True
	disc.compile(loss='binary_crossentropy', optimizer = d_optim)
	#vis_utils.plot_model(disc, to_file = '%s.png'%gan_model.name, show_shapes = True, show_layer_names = True)


	for epoch in range(n_epoch):
		print "Epoch : ", epoch
		batch = 0
		#real image batch
		miniBatch = createMinibatches(mbSize)
		imgBatch = miniBatch()

		for realImgBatch in imgBatch:
			print "Epoch : ", epoch, "Batch : ", batch
			batch += 1
			#generate fake image batch
			nzBatch = np.random.uniform(0, 1, size = [realImgBatch.shape[0], nzLen])
			fakeImgBatch = gen.predict(nzBatch)
			#plotGeneratedImage(fakeImgBatch)

			'''
			if batch % 20 == 0:
				image = combine_images(fakeImgBatch)
				image = image*std_+ mean_
				Image.fromarray(image.astype(np.uint8)).save(\
	           	 str(epoch)+"_"+str(batch)+".png")
	        '''

			#train discriminator
			trainData = np.concatenate((realImgBatch, fakeImgBatch))
			trainLabel = np.zeros([2 * realImgBatch.shape[0]])
			trainLabel[0 : realImgBatch.shape[0] - 1] = 0.9
			trainLabel[realImgBatch.shape[0] : ] = 0
			#print trainData.shape
			d_loss = disc.train_on_batch(trainData, trainLabel, sample_weight = None)
			loss["d"].append(d_loss)

			#train generator
			nzBatch = np.random.uniform(0, 1, size = [realImgBatch.shape[0], nzLen])
			trainLabel = np.zeros([realImgBatch.shape[0]])
			trainLabel[:] = 0.9
			g_loss =gan_model.train_on_batch(nzBatch, trainLabel)
			loss["g"].append(g_loss)
			
			print "Discriminator Loss : ", d_loss
			print "Generator Loss : ", g_loss

		nzBatch = np.random.uniform(0, 1, size = [16, nzLen])
		fakeImgBatch = gen.predict(nzBatch)
		image = combine_images(fakeImgBatch)
		#image = image * std_ + mean_
		image *= 0.5 
		image += 0.5 
		image *= 255.
		Image.fromarray(image.astype(np.uint8)).save(\
	           	 str(epoch)+"_"+".png")

def save_loss():
	fdisc = open("disc_loss", "wb")
	fgen = open("gen_loss", "wb")
	cPickle.dump(loss["d"], fdisc)
	cPickle.dump(loss["g"], fgen)
	fdisc.close()
	fgen.close()

def main():
	global img, loss
	img = read_data()
	#shp = img.shape[1:]
	#train()
	train(128, 50, 100)
	#plot_loss()
	#plot_gen()
	save_loss()
	

if __name__ == "__main__":
	main()