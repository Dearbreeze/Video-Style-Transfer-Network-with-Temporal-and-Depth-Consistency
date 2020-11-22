import numpy as np
import os
import sys
from PIL import Image
import cv2
import torch
import torch.nn as nn

def get_flow(filename):
	with open(filename, 'rb') as f:
		magic = np.fromfile(f, np.float32, count=1)
		if 202021.25 != magic:
			print('Magic number incorrect. Invalid .flo file')
		else:
			w = np.fromfile(f, np.int32, count=1)[0]
			h = np.fromfile(f, np.int32, count=1)[0]
			print('Reading %d x %d flo file' % (w, h))
			data = np.fromfile(f, np.float32, count=2*w*h)
			# Reshape data into 3D array (columns, rows, bands)
			data2D = np.resize(data, (1, h, w,2))
			data2D = np.transpose(data2D,[0, 3,1,2])
			return data2D

def get_pixel_value(img, x, y):
	shape = x.shape
	# x = x.permute(0,2,3,1)
	# img: tensor of shape (B, H, W, C)
	batch_size = shape[0]
	height = shape[1]
	width = shape[2]

	batch_idx = torch.arange(0, batch_size)
	batch_idx = torch.reshape(batch_idx, (batch_size, 1, 1))
	b = batch_idx.expand(1, height, width).int()
	indices = torch.stack([b, y, x], 3)
	#print(indices.shape)
	idx1, idx2, idx3 = indices.long().chunk(3, dim=3)
	return img[idx1, idx2, idx3].squeeze()

def warp(img, flow, H, W):
#   img batch,H,W,3
	B, H, W, C = img.size()
	xx = torch.arange(0, W).view(1,-1).repeat(H,1)
	yy = torch.arange(0, H).view(-1,1).repeat(1,W)
	xx = xx.view(1,1,H,W).repeat(B,1,1,1)
	yy = yy.view(1,1,H,W).repeat(B,1,1,1)
	#print(yy.shape)
	grid = torch.cat((xx,yy),1).float()
	#print(grid.shape)
	if img.is_cuda:
		grid = grid.cuda()
	flows = grid + flow

	max_y = int(H - 1)
	max_x = int(W - 1)


	x = flows[:,0,:,:]
	y = flows[:,1,:,:]
	x0 = x
	y0 = y
	x0 = x0.int()
	x1 = x0 + 1
	y0 = y0.int()
	y1 = y0 + 1

	# clip to range [0, H/W] to not violate img boundaries
	x0 = torch.clamp(x0, 0, max_x)
	x1 = torch.clamp(x1, 0, max_x)
	y0 = torch.clamp(y0, 0, max_y)
	y1 = torch.clamp(y1, 0, max_y)

	# get pixel value at corner coords
	Ia = get_pixel_value(img, x0, y0)
	Ib = get_pixel_value(img, x0, y1)
	Ic = get_pixel_value(img, x1, y0)
	Id = get_pixel_value(img, x1, y1)

	# recast as float for delta calculation
	x0 = x0.float()
	x1 = x1.float()
	y0 = y0.float()
	y1 = y1.float()


	# calculate deltas
	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	# add dimension for addition
	wa = wa.unsqueeze(3)
	wb = wb.unsqueeze(3)
	wc = wc.unsqueeze(3)
	wd = wd.unsqueeze(3)

	# compute output
	out = wa*Ia + wb*Ib + wc*Ic + wd*Id
	return out

