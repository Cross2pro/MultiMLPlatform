#!/usr/bin/env python
# -*- coding: utf-8 -*-

from app.models.model_manager import model_manager

print("开始测试模型加载...")
try:
    model_manager.initialize_models()
    print("模型加载成功!")
    
    # 测试随机森林模型
    rf_info = model_manager.models['randomForest'].get_model_info()
    print(f"随机森林模型信息: {rf_info}")
    
    # 测试OptimalNN模型
    if 'optimalNN' in model_manager.models:
        nn_info = model_manager.models['optimalNN'].get_model_info()
        print(f"OptimalNN模型信息: {nn_info}")
    
    # 测试可用模型列表
    available_models = model_manager.get_available_models()
    print(f"可用模型: {[model['name'] for model in available_models]}")
    
except Exception as e:
    print(f"测试失败: {str(e)}") 