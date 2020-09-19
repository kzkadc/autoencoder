# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import cuda,Function, Variable, optimizers, serializers, utils, iterators
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

def compose(x, funcs):
	y = x
	for f in funcs:
		y = f(y)
		
	return y

class AutoEncoder(Chain):
	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.fc1 = L.Linear(28*28, 1000)
			self.fc2 = L.Linear(1000, 800)
			self.fc3 = L.Linear(800, 500)
			self.fc4 = L.Linear(500, 100)
			self.fc5 = L.Linear(100, 10)
			self.fc6 = L.Linear(10, 2)
			
			self.out_bias = chainer.Parameter(0.0, [28*28])
		
	def __call__(self,x):
		_x = F.reshape(x, (-1,28*28))
		h = self.encode(_x)
		h = self.decode(h)
		loss = F.mean_squared_error(_x, h)
		
		return loss, h
		
	def encode(self, x):
		h = compose(x, [
			self.fc1, F.relu,
			self.fc2, F.relu,
			self.fc3, F.relu,
			self.fc4, F.relu,
			self.fc5, F.relu,
			self.fc6, F.relu
		])
		
		return h
		
	def decode(self, c):
		# encoderと重みを共有 (tied-weight)
		w6 = F.transpose(self.fc6.W)
		w5 = F.transpose(self.fc5.W)
		w4 = F.transpose(self.fc4.W)
		w3 = F.transpose(self.fc3.W)
		w2 = F.transpose(self.fc2.W)
		w1 = F.transpose(self.fc1.W)
		
		h = compose(c, [
			lambda x:F.linear(x, w6, self.fc5.b), F.relu,
			lambda x:F.linear(x, w5, self.fc4.b), F.relu,
			lambda x:F.linear(x, w4, self.fc3.b), F.relu,
			lambda x:F.linear(x, w3, self.fc2.b), F.relu,
			lambda x:F.linear(x, w2, self.fc1.b), F.relu,
			lambda x:F.linear(x, w1, self.out_bias), F.sigmoid
		])
		
		return h

class AE_CNN(Chain):
	def __init__(self):
		super().__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(1, 32, ksize=4, stride=2, pad=2,nobias=True)
			self.bne1 = L.BatchNormalization(32)
			self.conv2 = L.Convolution2D(32, 64, ksize=2, nobias=True)
			self.bne2 = L.BatchNormalization(64)
			self.conv3 = L.Convolution2D(64 ,64, ksize=2,stride=2, nobias=True)
			self.bne3 = L.BatchNormalization(64)
			self.conv4 = L.Convolution2D(64, 1, ksize=4)
			self.bne4 = L.BatchNormalization(1)
			
			self.bnd4 = L.BatchNormalization(64)
			self.bnd3 = L.BatchNormalization(64)
			self.bnd2 = L.BatchNormalization(32)
		
	def __call__(self, x):
		_x = F.reshape(x, (-1,1,28,28))
		h = self.encode(_x)
		h = self.decode(h)

		#h = F.reshape(h, (-1,28,28))
		loss = F.mean_squared_error(_x,h)
		return loss,h
		
	def encode(self,x):
		h = compose(x, [
			self.conv1, self.bne1, F.relu,
			self.conv2, self.bne2, F.relu,
			self.conv3, self.bne3, F.relu,
			self.conv4, self.bne4, F.relu
		])
		return h
		
	def decode(self, c):
		h = compose(c, [
			lambda x:F.deconvolution_2d(x, self.conv4.W),
			self.bnd4, F.relu,
			lambda x:F.deconvolution_2d(x, self.conv3.W, stride=2),
			self.bnd3, F.relu,
			lambda x:F.deconvolution_2d(x, self.conv2.W),
			self.bnd2, F.relu,
			lambda x:F.deconvolution_2d(x, self.conv1.W, stride=2, pad=2),
			F.sigmoid
		])
		return h
