#!/usr/bin/env python

import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import CrossEntropyLoss2d
from models import place_net
from scipy import ndimage
import matplotlib.pyplot as plt


class PlaceTrainer(object):
    """放置训练器：专门用于训练放置位置的自监督学习"""
    
    def __init__(self, is_testing, load_snapshot, snapshot_file, force_cpu):
        self.is_testing = is_testing
        
        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Initialize placement network
        self.model = place_net(self.use_cuda)

        # Initialize classification loss for placement
        place_num_classes = 3  # 0 - 成功放置, 1 - 放置失败, 2 - 无损失
        place_class_weights = torch.ones(place_num_classes)
        place_class_weights[place_num_classes - 1] = 0  # 忽略"无损失"类别
        if self.use_cuda:
            self.place_criterion = CrossEntropyLoss2d(place_class_weights.cuda()).cuda()
        else:
            self.place_criterion = CrossEntropyLoss2d(place_class_weights)

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained placement model snapshot loaded from: %s' % (snapshot_file))

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)
        self.iteration = 0

        # Initialize lists to save execution info
        self.executed_place_log = []
        self.place_label_value_log = []
        self.place_predicted_value_log = []

    def forward(self, color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=-1):
        """前向传播：输入场景图像，输出放置位置的概率图"""
        
        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2,2,1], order=0)
        depth_heightmap_2x = ndimage.zoom(depth_heightmap, zoom=[2,2], order=0)
        assert(color_heightmap_2x.shape[0:2] == depth_heightmap_2x.shape[0:2])

        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length/32)*32
        padding_width = int((diag_length - color_heightmap_2x.shape[0])/2)
        color_heightmap_2x_r =  np.pad(color_heightmap_2x[:,:,0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g =  np.pad(color_heightmap_2x[:,:,1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b =  np.pad(color_heightmap_2x[:,:,2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)
        depth_heightmap_2x =  np.pad(depth_heightmap_2x, padding_width, 'constant', constant_values=0)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float)/255
        for c in range(3):
            input_color_image[:,:,c] = (input_color_image[:,:,c] - image_mean[c])/image_std[c]

        # Pre-process depth image (normalize)
        image_mean = [0.01, 0.01, 0.01]
        image_std = [0.03, 0.03, 0.03]
        depth_heightmap_2x.shape = (depth_heightmap_2x.shape[0], depth_heightmap_2x.shape[1], 1)
        input_depth_image = np.concatenate((depth_heightmap_2x, depth_heightmap_2x, depth_heightmap_2x), axis=2)
        for c in range(3):
            input_depth_image[:,:,c] = (input_depth_image[:,:,c] - image_mean[c])/image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_depth_image.shape = (input_depth_image.shape[0], input_depth_image.shape[1], input_depth_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3,2,0,1)
        input_depth_data = torch.from_numpy(input_depth_image.astype(np.float32)).permute(3,2,0,1)

        # Pass input data through model
        output_prob, state_feat = self.model.forward(input_color_data, input_depth_data, is_volatile, specific_rotation)

        # Return placement predictions (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                place_predictions = F.softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[:,0,(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]
            else:
                place_predictions = np.concatenate((place_predictions, F.softmax(output_prob[rotate_idx][0], dim=1).cpu().data.numpy()[:,0,(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2),(padding_width/2):(color_heightmap_2x.shape[0]/2 - padding_width/2)]), axis=0)

        return place_predictions, state_feat

    def get_place_label_value(self, place_success, stability_score):
        """根据放置结果计算标签值"""
        
        # 计算标签值
        label_value = 0
        if not place_success:
            label_value = 1  # 放置失败
        elif stability_score < 0.5:  # 稳定性评分较低
            label_value = 1  # 不稳定放置也算失败
            
        print('Place label value: %d' % (label_value))
        return label_value

    def backprop(self, color_heightmap, depth_heightmap, best_pix_ind, label_value):
        """反向传播，更新模型权重"""
        
        # Compute fill value
        fill_value = 2

        # Compute labels
        label = np.zeros((1,320,320)) + fill_value
        action_area = np.zeros((224,224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        tmp_label = np.zeros((224,224)) + fill_value
        tmp_label[action_area > 0] = label_value
        label[0,48:(320-48),48:(320-48)] = tmp_label

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        
        # Do forward pass with specified rotation (to save gradients)
        place_predictions, state_feat = self.forward(color_heightmap, depth_heightmap, is_volatile=False, specific_rotation=best_pix_ind[0])

        if self.use_cuda:
            loss = self.place_criterion(self.model.output_prob[0][0], Variable(torch.from_numpy(label).long().cuda()))
        else:
            loss = self.place_criterion(self.model.output_prob[0][0], Variable(torch.from_numpy(label).long()))
        loss.backward()
        loss_value = loss.cpu().data.numpy()

        print('Placement training loss: %f' % (loss_value))
        self.optimizer.step()

    def place_heuristic(self, depth_heightmap):
        """启发式放置策略：寻找合适的放置位置"""
        
        num_rotations = 16
        for rotate_idx in range(num_rotations):
            rotated_heightmap = ndimage.rotate(depth_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            valid_areas = np.zeros(rotated_heightmap.shape)
            
            # 寻找平坦区域作为放置位置
            # 计算局部高度变化
            kernel_size = 15
            local_std = ndimage.generic_filter(rotated_heightmap, np.std, size=kernel_size)
            # 平坦区域：高度变化小
            valid_areas[local_std < 0.01] = 1
            
            # 确保有足够的支撑面积
            min_area = 100  # 最小支撑面积
            from scipy import ndimage
            labeled, num_features = ndimage.label(valid_areas)
            for i in range(1, num_features + 1):
                if np.sum(labeled == i) < min_area:
                    valid_areas[labeled == i] = 0

            # 模糊处理
            blur_kernel = np.ones((25,25),np.float32)/9
            valid_areas = cv2.filter2D(valid_areas, -1, blur_kernel)
            tmp_place_predictions = ndimage.rotate(valid_areas, -rotate_idx*(360.0/num_rotations), reshape=False, order=0)
            tmp_place_predictions.shape = (1, rotated_heightmap.shape[0], rotated_heightmap.shape[1])
            
            if rotate_idx == 0:
                place_predictions = tmp_place_predictions
            else:
                place_predictions = np.concatenate((place_predictions, tmp_place_predictions), axis=0)

        best_pix_ind = np.unravel_index(np.argmax(place_predictions), place_predictions.shape)
        return best_pix_ind

    def get_prediction_vis(self, predictions, color_heightmap, best_pix_ind):
        """可视化放置预测结果"""
        
        canvas = None
        num_rotations = predictions.shape[0]
        for canvas_row in range(int(num_rotations/4)):
            tmp_row_canvas = None
            for canvas_col in range(4):
                rotate_idx = canvas_row*4+canvas_col
                prediction_vis = predictions[rotate_idx,:,:].copy()
                prediction_vis = np.clip(prediction_vis, 0, 1)
                prediction_vis.shape = (predictions.shape[1], predictions.shape[2])
                prediction_vis = cv2.applyColorMap((prediction_vis*255).astype(np.uint8), cv2.COLORMAP_JET)
                if rotate_idx == best_pix_ind[0]:
                    prediction_vis = cv2.circle(prediction_vis, (int(best_pix_ind[2]), int(best_pix_ind[1])), 7, (0,0,255), 2)
                prediction_vis = ndimage.rotate(prediction_vis, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                background_image = ndimage.rotate(color_heightmap, rotate_idx*(360.0/num_rotations), reshape=False, order=0)
                prediction_vis = (0.5*cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR) + 0.5*prediction_vis).astype(np.uint8)
                if tmp_row_canvas is None:
                    tmp_row_canvas = prediction_vis
                else:
                    tmp_row_canvas = np.concatenate((tmp_row_canvas,prediction_vis), axis=1)
            if canvas is None:
                canvas = tmp_row_canvas
            else:
                canvas = np.concatenate((canvas,tmp_row_canvas), axis=0)

        return canvas 