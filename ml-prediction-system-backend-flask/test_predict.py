#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.models.model_manager import model_manager

# 测试特征数据
test_features = {
    "joint_type": 1,
    "specimen_type": 1,
    "key_number": 3,
    "key_width": 50,
    "key_root_height": 25,
    "key_depth": 15,
    "key_inclination": 90,
    "key_spacing": 100,
    "key_front_height": 25,
    "key_depth_height_ratio": 0.6,
    "joint_width": 150,
    "joint_height": 300,
    "key_area": 750,
    "joint_area": 45000,
    "flat_region_area": 42750,
    "key_joint_area_ratio": 0.0167,
    "compressive_strength": 40,
    "fiber_type": 1,
    "fiber_volume_fraction": 0.01,
    "fiber_length": 30,
    "fiber_diameter": 0.5,
    "fiber_reinforcing_index": 0.6,
    "confining_stress": 1.0,
    "confining_ratio": 0.025
}

print("====== 测试随机森林模型预测 ======")
try:
    rf_result = model_manager.predict('randomForest', test_features)
    print(f"随机森林预测结果: {rf_result}")
except Exception as e:
    print(f"随机森林预测失败: {str(e)}")

print("\n====== 测试OptimalNN模型预测 ======")
try:
    nn_result = model_manager.predict('optimalNN', test_features)
    print(f"OptimalNN预测结果: {nn_result}")
except Exception as e:
    print(f"OptimalNN预测失败: {str(e)}")

print("\n====== 比较两种模型的预测结果 ======")
try:
    if 'shear_capacity' in rf_result and 'shear_capacity' in nn_result:
        rf_value = rf_result['shear_capacity']
        nn_value = nn_result['shear_capacity']
        diff = abs(rf_value - nn_value)
        relative_diff = diff / abs(rf_value) if rf_value != 0 else 0
        
        print(f"随机森林预测值: {rf_value}")
        print(f"OptimalNN预测值: {nn_value}")
        print(f"绝对差异: {diff}")
        print(f"相对差异: {relative_diff * 100:.2f}%")
except Exception as e:
    print(f"比较失败: {str(e)}") 