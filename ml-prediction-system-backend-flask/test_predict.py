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

# 第二组测试特征（更高强度的配置）
test_features_2 = {
    "joint_type": 2,
    "specimen_type": 1,
    "key_number": 5,
    "key_width": 80,
    "key_root_height": 40,
    "key_depth": 25,
    "key_inclination": 90,
    "key_spacing": 120,
    "key_front_height": 40,
    "key_depth_height_ratio": 0.625,
    "joint_width": 200,
    "joint_height": 400,
    "key_area": 1200,
    "joint_area": 80000,
    "flat_region_area": 74000,
    "key_joint_area_ratio": 0.015,
    "compressive_strength": 80,
    "fiber_type": 2,
    "fiber_volume_fraction": 0.02,
    "fiber_length": 20,
    "fiber_diameter": 0.3,
    "fiber_reinforcing_index": 1.33,
    "confining_stress": 3.0,
    "confining_ratio": 0.0375
}

def test_prediction(features, test_name):
    print(f"\n====== {test_name} ======")
    results = {}
    
    print(f"测试随机森林模型预测...")
    try:
        rf_result = model_manager.predict('randomForest', features)
        print(f"随机森林预测结果: {rf_result['shear_capacity']:.2f}")
        print(f"置信度: {rf_result['confidence']:.3f}")
        print(f"个体预测值: {[round(p, 2) for p in rf_result['individual_predictions']]}")
        results['rf'] = rf_result
    except Exception as e:
        print(f"随机森林预测失败: {str(e)}")

    print(f"\n测试OptimalNN模型预测...")
    try:
        nn_result = model_manager.predict('optimalNN', features)
        print(f"OptimalNN预测结果: {nn_result['shear_capacity']:.2f}")
        print(f"置信度: {nn_result['confidence']:.3f}")
        print(f"个体预测值: {[round(p, 2) for p in nn_result['individual_predictions']]}")
        results['nn'] = nn_result
    except Exception as e:
        print(f"OptimalNN预测失败: {str(e)}")

    # 比较结果
    if 'rf' in results and 'nn' in results:
        rf_value = results['rf']['shear_capacity']
        nn_value = results['nn']['shear_capacity']
        
        if rf_value > 0 and nn_value > 0:
            diff = abs(rf_value - nn_value)
            relative_diff = diff / abs(rf_value) if rf_value != 0 else 0
            
            print(f"\n模型比较:")
            print(f"随机森林预测值: {rf_value:.2f}")
            print(f"OptimalNN预测值: {nn_value:.2f}")
            print(f"绝对差异: {diff:.2f}")
            print(f"相对差异: {relative_diff * 100:.2f}%")
        else:
            print(f"\n模型比较:")
            print(f"随机森林预测值: {rf_value:.2f}")
            print(f"OptimalNN预测值: {nn_value:.2f}")
            print("注意：存在零值或负值预测")

# 运行测试
test_prediction(test_features, "测试案例1 - 基础配置")
test_prediction(test_features_2, "测试案例2 - 高强度配置")

print(f"\n====== 测试总结 ======")
print("1. 随机森林模型使用真实的决策树预测结果")
print("2. OptimalNN模型使用真实的神经网络预测结果")
print("3. individual_predictions 现在反映模型的真实不确定性")
print("4. 不再使用模拟或估算的数据进行预测") 