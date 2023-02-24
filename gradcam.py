import os


os.environ['CUDA_VISIBLE_DEVICES'] = '3'


import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torchvision import models
import resnet
import pdb
import torchvision
import torchvision.io
import imutils

device = torch.device("cuda")
class FeatureExtractor():
	""" Class for extracting activations and
	registering gradients from targetted intermediate layers """

	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []

		x = self.model(x)
		x.register_hook(self.save_gradient)
		outputs += [x]
		#pdb.set_trace()
		'''
		for name, module in self.model._modules.items():
			x = module(x)
			if name in self.target_layers:
				x.register_hook(self.save_gradient)
				outputs += [x]
		'''
		return outputs, x


class ModelOutputs():
	""" Class for making a forward pass, and getting:
	1. The network output.
	2. Activations from intermeddiate targetted layers.
	3. Gradients from intermeddiate targetted layers. """

	def __init__(self, model, feature_module, target_layers):
		self.model = model
		self.feature_module = feature_module
		self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations = []
		for name, module in self.model._modules.items():
			if module == self.feature_module:
				target_activations, x = self.feature_extractor(x)
			elif "avgpool" in name.lower():
				x = module(x)
				x = x.view(x.size(0),-1)
			else:
				x = module(x)

		return target_activations, x

def show_cam_on_image(img, mask, ret1, ret2, fname='cam.jpg'):
	#heatmap = np.uint8(255 * mask)
	heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
	ret1.append(heatmap.copy())
	#cv2.imwrite(fname.replace('.jpg', '_g.jpg'), heatmap)
	heatmap = np.float32(heatmap) #0-255


	mean = [0.4345, 0.4051, 0.3775]
	std = [0.2768, 0.2713, 0.2737]

	for z in range(3):
		img[:, :, z] = img[:, :, z] * std[z] + mean[z]

	img *= 255.0
	cam = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 0-255
	ret2.append(cam.copy())
	#cv2.imwrite(fname.replace('.jpg', '_i.jpg'), cam)
	#cam -= np.min(cam)
	#cam /= np.max(cam)

	#pdb.set_trace()
	#cam *= np.expand_dims(mask, 2)
	#cam += heatmap #0-510
	#cam /= 2
	#cam = cam / np.max(cam)
	#cv2.imwrite(fname, cam)
	'''
	cam = np.float32(img)
	cam -= np.min(cam)
	cam /= np.max(cam)
	cv2.imwrite(fname, np.uint8(255 * cam))
	'''

class GradCam:
	def __init__(self, model, feature_module, target_layer_names, use_cuda):
		self.model = model
		self.feature_module = feature_module
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.cuda()

		self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index=None):
		if self.cuda:
			features, output = self.extractor(input.cuda())
		else:
			features, output = self.extractor(input)

		if index == None:
			index = np.argmax(output.cpu().data.numpy())

		one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot).requires_grad_(True)
		if self.cuda:
			one_hot = torch.sum(one_hot.cuda() * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.feature_module.zero_grad()
		self.model.zero_grad()
		one_hot.backward(retain_graph=True)

		grads_val_all = self.extractor.get_gradients()[-1].cpu().data.numpy()
		target_all = features[-1]
		target_all = target_all.cpu().data.numpy()[0, :]

		cams = []
		for i in range(16):
			grads_val = grads_val_all[:, :, i, :, :]
			target = target_all[:, i, :, :]

			weights = np.mean(grads_val, axis=(2, 3))[0, :]
			cam = np.zeros(target.shape[1:], dtype=np.float32)

			for i, w in enumerate(weights):
				cam += w * target[i, :, :]

			#cam = np.maximum(cam, 0)
			cam = cv2.resize(cam, (112, 112))
			#cam = cam - np.min(cam)
			#cam = cam / np.max(cam)

			cams.append(cam)

		return np.asarray(cams)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--use-cuda', action='store_true', default=False,
						help='Use NVIDIA GPU acceleration')
	parser.add_argument('--image-path', type=str, default='./examples/both.png',
						help='Input image path')
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()
	if args.use_cuda:
		print("Using GPU for acceleration")
	else:
		print("Using CPU for computation")

	return args

def deprocess_image(img):
	""" see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
	img = img - np.mean(img)
	img = img / (np.std(img) + 1e-5)
	img = img * 0.1
	img = img + 0.5
	img = np.clip(img, 0, 1)
	return np.uint8(img*255)


if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	args = get_args()

	# Can work with any model, but it assumes that the model has a
	# feature method, and a classifier method,
	# as in the VGG models in torchvision.
	#model = models.resnet50(pretrained=True)

	pretrain = torch.load('model/r3d18_K_200ep.pth', map_location='cuda')
	model = resnet.generate_model(model_depth=18, n_classes=700)
	model.load_state_dict(pretrain['state_dict'])
	model.fc = nn.Linear(model.fc.in_features, 3)
	#load 20
	model_path = f'model/exa_resnet_18_xentropy_epoch_13.pth'
	model.load_state_dict(torch.load(model_path))

	freeze = True
	if freeze:
		model_freeze = resnet.generate_model(model_depth=18, n_classes=700)
		model_freeze.load_state_dict(pretrain['state_dict'])

		model.conv1.weight = model_freeze.conv1.weight
		model.bn1.weight = model_freeze.bn1.weight
		model.bn1.bias = model_freeze.bn1.bias

	model.eval()

	grad_cam = GradCam(model=model, feature_module=model.conv1, \
					   target_layer_names=["1"], use_cuda=True)

	fname = os.listdir('cache_test')
	for f in fname:
		path = 'cache_test/' + f
		X_all = torchvision.io.read_video(path, pts_unit='sec')[0]
		n = len(X_all)
		#n = 800
		#pdb.set_trace()
		for i in range(0, n-16):
			if i > 1000:
				break
			#if i < 1600:
			#	continue
			#
			#i = i_end - 16
			i_end = i + 16
			X = X_all[i:i_end]
			X = torch.unsqueeze(X, 0)
			X = X.to(device)
			X = (X / 255.0)
			mean = [0.4345, 0.4051, 0.3775]
			std = [0.2768, 0.2713, 0.2737]

			for z in range(3):
				X[:, :, :, :, z] = (X[:, :, :, :, z] - mean[z]) / std[z]

			X = X.permute(0, 4, 1, 2, 3)

			# If None, returns the map for the highest scoring category.
			# Otherwise, targets the requested index.
			target_index = 2
			mask = np.abs(grad_cam(X, target_index))
			mask /= 0.0005
			#mask /= np.max(mask)
			mask = np.clip(mask, 0, 1)

			ret1 = []
			ret2 = []
			os.makedirs(f'cam3_end/{f}', exist_ok=True)
			for j in range(16):
				img = X[0, :, j, :, :]
				img = img.permute(1, 2, 0).cpu().numpy()

				show_cam_on_image(img, mask[j], ret1, ret2, fname=f'cam3_end/cam_{i_end}/{j}.jpg')
			montages  = imutils.build_montages(ret1 + ret2, (112, 112), (16, 2))
			cv2.imwrite(f'cam3_end/{f}/cam_{i}.jpg', montages[0])
			print(i)
	'''
	gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
	print(model._modules.items())
	gb = gb_model(input, index=target_index)
	gb = gb.transpose((1, 2, 0))
	cam_mask = cv2.merge([mask, mask, mask])
	cam_gb = deprocess_image(cam_mask*gb)
	gb = deprocess_image(gb)

	cv2.imwrite('gb.jpg', gb)
	cv2.imwrite('cam_gb.jpg', cam_gb)
	'''
