import os
from app.models.random_forest import RandomForestModel
from app.models.optimal_nn import OptimalNNModel
from app.utils.model_utils import ModelUtils

class ModelManager:
    def __init__(self):
        self.models = {
            'randomForest': RandomForestModel(),
            'optimalNN': OptimalNNModel()
        }
        
        self.model_info = {
            'randomForest': {
                'name': '随机森林',
                'description': '基于多个决策树的集成学习模型',
                'trained': False,
                'accuracy': 0.934,
                'metrics': None,
                'predictTime': 0
            },
            'optimalNN': {
                'name': '最优神经网络',
                'description': '基于PyTorch的优化神经网络模型',
                'trained': False,
                'accuracy': 0.893,
                'metrics': None,
                'predictTime': 0
            }
        }
        
        # 初始化时加载模型
        self.initialize_models()
    
    def initialize_models(self):
        """初始化并加载所有预训练模型"""
        try:
            # 加载随机森林模型
            self.load_model('randomForest')
            
            # 加载最优神经网络模型
            try:
                self.load_model('optimalNN')
            except Exception as e:
                print(f'加载OptimalNN模型失败: {str(e)}')
        except Exception as e:
            print(f'初始化模型失败: {str(e)}')
    
    def get_available_models(self):
        """
        获取所有可用模型的信息
        :return: 模型信息数组
        """
        return [{'id': key, **info} for key, info in self.model_info.items()]
    
    def load_model(self, model_type, model_path=None):
        """
        加载指定类型的预训练模型
        :param model_type: 模型类型
        :param model_path: 模型路径（可选）
        :return: 加载结果和模型信息
        """
        if model_type not in self.models:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        print(f"开始加载{self.model_info[model_type]['name']}模型...")
        
        try:
            # 加载模型
            self.models[model_type].load_model(model_path)
            
            # 更新模型信息
            self.model_info[model_type]['trained'] = True
            
            # 获取模型信息
            model_info = self.models[model_type].get_model_info()
            
            return {
                'modelType': model_type,
                'modelName': self.model_info[model_type]['name'],
                'trained': True,
                **model_info
            }
        except Exception as e:
            print(f"加载{model_type}模型失败: {str(e)}")
            raise e
    
    def predict(self, model_type, features):
        """
        使用指定模型进行预测
        :param model_type: 模型类型
        :param features: 输入特征
        :return: 预测结果
        """
        if model_type not in self.models:
            raise ValueError(f"不支持的模型类型: {model_type}")
        
        if not self.model_info[model_type]['trained']:
            # 如果模型尚未加载，尝试加载
            self.load_model(model_type)
        
        # 记录预测开始时间
        import time
        start_time = time.time()
        
        # 进行预测
        prediction = self.models[model_type].predict(features)
        
        # 计算预测用时
        predict_time = int((time.time() - start_time) * 1000)
        
        return {
            **prediction,
            'modelName': self.model_info[model_type]['name'],
            'processTime': f"{predict_time}ms"
        }
    
    def train_model(self, model_type):
        """
        兼容旧接口，实际是加载预训练模型
        :param model_type: 模型类型
        :return: 加载结果和模型信息
        """
        return self.load_model(model_type)
    
    def evaluate_predict(self, model_type, actual_value, predicted_value):
        """
        评估模型预测效果
        :param model_type: 模型类型
        :param actual_value: 实际值
        :param predicted_value: 预测值
        :return: 评估结果
        """
        error = abs(actual_value - predicted_value)
        relative_error = error / abs(actual_value) if actual_value != 0 else 1
        
        return {
            'actual': actual_value,
            'predicted': predicted_value,
            'error': error,
            'relativeError': f"{relative_error * 100}%",
            'accuracy': f"{(1 - relative_error) * 100}%"
        }

# 创建单例实例
model_manager = ModelManager() 