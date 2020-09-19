# -*- coding: utf-8 -*-

import numpy as np
import chainer
from chainer import Function, Variable, optimizers, serializers, utils, iterators
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import computational_graph
from chainer.backends import cuda

import cv2
import random, pprint
from pathlib import Path


def parse_args():
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("-b", "--batchsize", type=int, default=50, help="batchsize")
	parser.add_argument("-e", "--epoch", type=int, default=100, help="iteration")
	parser.add_argument("-g", "--gpu", type=int, default=-1, help="GPU ID")
	parser.add_argument("--graph", default=None, help="computational graph")
	parser.add_argument("--cnn", action="store_true", help="CNN")
	parser.add_argument("--lam", type=float, default=0.0, help="weight decay")
	parser.add_argument("-m", "--model", default="autoencoder.model", help="model file name")
	parser.add_argument("-r", "--result", default="result", help="result directory")
	args = parser.parse_args()
	
	pprint.pprint(vars(args))
	main(args)

	
def main(args):
	# GPUを使うための設定
	chainer.config.user_gpu = args.gpu
	if args.gpu >= 0:
		print("GPU MODE")
		cuda.get_device_from_id(args.gpu).use()
	else:
		print("CPU MODE")
		
	from model import AutoEncoder, AE_CNN
	
	# モデル作成
	# CNNか全結合モデルか引数で選べるようにしてある
	if args.cnn:
		autoencoder = AE_CNN()
		print("CNN model")
	else:
		autoencoder = AutoEncoder()
		print("Fully-connected model")
	
	if args.gpu >= 0:
		autoencoder.to_gpu()
	
	# optimizer作成
	optimizer = optimizers.Adam()
	optimizer.setup(autoencoder)
	optimizer.add_hook(chainer.optimizer.WeightDecay(args.lam))
	
	# データを読み込み
	train, test = chainer.datasets.get_mnist(withlabel=False, ndim=2)
	
	# イテレータを作る
	BATCH_SIZE = args.batchsize
	train_iter = iterators.SerialIterator(train, BATCH_SIZE)
	
	saver = MNISTSampleSaver(autoencoder, test, 10, args.result)
	saver.save_images("before")

    # 学習ループ
	i = 0
	while train_iter.epoch < args.epoch:
		i+=1
	
	    # バッチ取得
		x = train_iter.next()
		x = Variable(np.stack(x))
		if chainer.config.user_gpu >= 0:
			x.to_gpu()
		
		# 再構成損失を計算
		autoencoder.cleargrads()
		loss = autoencoder(x)[0]
		loss.backward()
		optimizer.update()
		
		if i == 1 and args.graph is not None:
			with open(args.graph,"w") as o:
				variable_style = {'shape': 'octagon', 'fillcolor': '#E0E0E0','style': 'filled'}
				function_style = {'shape': 'record', 'fillcolor': '#6495ED','style': 'filled'}
				g = computational_graph.build_computational_graph([loss], variable_style=variable_style, function_style=function_style)
				# lossを渡すとそこに至る計算グラフ(dot)が生成される
				o.write(g.dump())
			print('graph generated')
			gf = True

		if i == 1 or train_iter.is_new_epoch:
			serializers.save_npz(str(Path(args.result, args.model)), autoencoder)
			print("{:d}: Train loss:{}".format(train_iter.epoch, loss.array))
	
	
	saver.save_images("after")

    # テスト精度を算出
	print("Test phase")
	losses = []
	test_iter = iterators.SerialIterator(test, BATCH_SIZE, repeat=False, shuffle=False)
	while True:
		try:
			x = test_iter.next()
		except StopIteration:
			break
			
		x = Variable(np.stack(x))
		if chainer.config.user_gpu >= 0:
			x.to_gpu()
		
		loss = autoencoder(x)[0].array
		losses.append(autoencoder.xp.asnumpy(loss))
		
	print("Test loss:{}".format(np.mean(losses)))
	
	
class MNISTSampleSaver:
	def __init__(self, model, dataset, num, outdir):
		self.outdir_path = Path(outdir)
		try:
			self.outdir_path.mkdir(parents=True)
		except FileExistsError:
			pass
		
		self.model = model
		self.dataset = dataset
		self.indices = np.random.choice(len(dataset), size=num, replace=False)
		
		# 選んだ画像を保存
		samples = (self.dataset[self.indices]*255).astype(np.uint8).squeeze()
		assert len(samples.shape) == 3, samples.shape
		for i,img in enumerate(samples):
			p = self.outdir_path / "input_{:02d}.png".format(i)
			cv2.imwrite(str(p), img)
		
	def save_images(self, kw):
		samples = self.dataset[self.indices].reshape((-1,28,28))
		
		temp = Variable(samples)
		if chainer.config.user_gpu >= 0:
			temp.to_gpu()
		with chainer.using_config("train", False):
			sample_out = self.model(temp)[1].array*255
		
		sample_out = sample_out.astype(np.uint8).reshape((-1,28,28)).squeeze()
		if chainer.config.user_gpu >= 0:
			sample_out = self.model.xp.asnumpy(sample_out)
		
		assert len(sample_out.shape) == 3, sample_out.shape
		# 再構成画像を保存
		for i,img in enumerate(sample_out):
			p = self.outdir_path / "out_{}_{:02d}.png".format(kw, i)
			cv2.imwrite(str(p), img)
	
if __name__ == "__main__":
	parse_args()
	