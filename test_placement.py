#!/usr/bin/env python

import numpy as np
import torch
from trainer import Trainer
from robot import Robot
import time


def test_placement_system():
    """测试放置系统是否正常工作"""
    
    print("开始测试放置系统...")
    
    # 初始化参数
    method = 'reactive'
    is_testing = False
    load_snapshot = False
    snapshot_file = None
    force_cpu = True  # 强制使用CPU避免CUDA问题
    
    try:
        # 初始化训练器
        print("初始化训练器...")
        trainer = Trainer(method, None, None, is_testing, load_snapshot, snapshot_file, force_cpu)
        print("训练器初始化成功")
        
        # 初始化机器人（仿真模式）
        print("初始化机器人...")
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
        robot = Robot(True, 'objects/blocks', 3, workspace_limits, None, None, None, None, False, None, None)
        print("机器人初始化成功")
        
        # 重置环境
        print("重置仿真环境...")
        robot.restart_sim()
        robot.add_objects()
        
        # 获取场景图像
        print("获取相机数据...")
        color_img, depth_img = robot.get_camera_data()
        if color_img is None or depth_img is None:
            print("无法获取相机数据")
            return False
        
        print(f"图像尺寸: 彩色图 {color_img.shape}, 深度图 {depth_img.shape}")
        
        # 测试放置网络前向传播
        print("测试放置网络前向传播...")
        try:
            place_predictions, _ = trainer.forward_place(color_img, depth_img, is_volatile=True)
            print(f"放置预测成功，预测图尺寸: {place_predictions.shape}")
            
            # 选择最佳放置位置
            best_pix_ind = np.unravel_index(np.argmax(place_predictions), place_predictions.shape)
            predicted_value = np.max(place_predictions)
            print(f"最佳放置位置: ({best_pix_ind[0]}, {best_pix_ind[1]}, {best_pix_ind[2]})")
            print(f"预测置信度: {predicted_value:.3f}")
            
        except Exception as e:
            print(f"放置网络前向传播失败: {e}")
            return False
        
        # 测试启发式放置
        print("测试启发式放置...")
        try:
            place_position = robot.get_heuristic_place_position(depth_img, workspace_limits)
            if place_position is not None:
                print(f"启发式放置位置: {place_position}")
            else:
                print("启发式放置位置计算失败")
        except Exception as e:
            print(f"启发式放置失败: {e}")
        
        # 测试放置动作执行
        print("测试放置动作执行...")
        try:
            test_position = [0.0, 0.0, 0.1]  # 测试位置
            success = robot.place_object(test_position)
            print(f"放置动作执行: {'成功' if success else '失败'}")
        except Exception as e:
            print(f"放置动作执行失败: {e}")
        
        print("所有测试完成！")
        return True
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        return False


if __name__ == "__main__":
    success = test_placement_system()
    if success:
        print("✅ 放置系统测试通过！")
    else:
        print("❌ 放置系统测试失败！") 