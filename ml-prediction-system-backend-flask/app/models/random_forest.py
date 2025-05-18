import os
import pickle
import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self):
        self.name = '随机森林'
        self.trained = False
        self.model = None
        self.feature_names = [
            'joint_type', 'specimen_type', 'key_number', 
            'key_width', 'key_root_height', 'key_depth', 
            'key_inclination', 'key_spacing', 'key_front_height', 
            'key_depth_height_ratio', 'joint_width', 'joint_height', 
            'key_area', 'joint_area', 'flat_region_area', 
            'key_joint_area_ratio', 'compressive_strength', 'fiber_type', 
            'fiber_volume_fraction', 'fiber_length', 'fiber_diameter', 
            'fiber_reinforcing_index', 'confining_stress', 'confining_ratio'
        ]
        self.scaler = None
        # 自定义树模型
        self.custom_trees = None
        self.n_estimators = 0

    def load_model(self, model_path=None):
        """
        加载预训练模型
        :param model_path: 模型路径，不提供则使用默认路径
        :return: 是否加载成功
        """
        try:
            # 首先尝试加载JSON格式的简化模型
            default_json_path = os.path.join(os.path.dirname(__file__), '../data/rf_model_simplified.json')
            if model_path:
                file_path = model_path
            elif os.path.exists(default_json_path):
                file_path = default_json_path
            else:
                default_pkl_path = os.path.join(os.path.dirname(__file__), '../data/random_forest_model.pkl')
                file_path = default_pkl_path
            
            # 检查文件是否存在
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"模型文件不存在: {file_path}")
            
            # 根据文件扩展名选择加载方式
            if file_path.endswith('.json'):
                self._load_json_model(file_path)
            else:
                self._load_pickle_model(file_path)
            
            self.trained = True
            print('随机森林模型加载成功!')
            return True
        except Exception as e:
            print(f'加载模型失败: {str(e)}')
            raise Exception(f"加载模型失败: {str(e)}")

    def _load_pickle_model(self, file_path):
        """加载pickle格式的模型"""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.feature_names = model_data.get('feature_names', self.feature_names)

    def _load_json_model(self, file_path):
        """加载JSON格式的简化模型"""
        with open(file_path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
            # 提取特征名称
            if 'feature_names' in model_data:
                self.feature_names = model_data['feature_names']
            
            # 提取缩放器信息
            if 'scaler' in model_data:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.mean_ = np.array(model_data['scaler']['mean'])
                scaler.scale_ = np.array(model_data['scaler']['scale'])
                self.scaler = scaler
            
            # 提取树模型
            if 'trees' in model_data:
                self.custom_trees = model_data['trees']
                self.n_estimators = model_data.get('n_estimators', len(self.custom_trees))
                # 不创建sklearn的随机森林模型，使用自定义树结构
                self.model = None
            else:
                raise ValueError("JSON模型中缺少树结构信息")

    def predict(self, features):
        """
        使用预训练随机森林模型进行预测
        :param features: 输入特征
        :return: 预测结果
        """
        # 如果模型尚未加载，则加载模型
        if not self.trained:
            self.load_model()

        # 准备特征数据
        feature_vector = self._prepare_features(features)
        
        # 判断使用哪种方式进行预测
        if self.model is not None:
            # 使用sklearn模型进行预测
            result = float(self.model.predict([feature_vector])[0])
            
            # 获取各棵树的预测
            individual_predictions = []
            for tree in self.model.estimators_:
                prediction = float(tree.predict([feature_vector])[0])
                individual_predictions.append(prediction)
        else:
            # 使用自定义树模型进行预测
            individual_predictions = []
            for tree in self.custom_trees:
                prediction = self._predict_with_custom_tree(tree, feature_vector)
                individual_predictions.append(prediction)
            
            # 计算平均值作为最终预测结果
            result = sum(individual_predictions) / len(individual_predictions)
        
        return {
            'shear_capacity': result,
            'individual_predictions': individual_predictions,
            'confidence': self.calculate_confidence(individual_predictions)
        }

    def _predict_with_custom_tree(self, tree, features):
        """
        使用自定义决策树进行预测
        :param tree: 树结构
        :param features: 特征向量
        :return: 预测结果
        """
        # 从根节点开始遍历
        current_node = 0
        nodes = tree['nodes']
        
        while True:
            node = nodes[current_node]
            
            # 如果是叶节点，返回值
            if node['type'] == 'leaf':
                return float(node['value'])
            
            # 如果是内部节点，根据特征值和阈值决定走左子树还是右子树
            feature_idx = node['feature']
            threshold = node['threshold']
            
            if features[feature_idx] <= threshold:
                current_node = node['left_child']
            else:
                current_node = node['right_child']

    def _prepare_features(self, features):
        """
        准备特征向量
        :param features: 输入特征字典
        :return: 特征向量
        """
        # 创建特征向量
        feature_vector = []
        for feature_name in self.feature_names:
            feature_value = features.get(feature_name, 0)
            feature_vector.append(feature_value)
        
        # 应用缩放器（如果存在）
        if self.scaler:
            feature_vector = self.scaler.transform([feature_vector])[0]
            
        return feature_vector

    def calculate_confidence(self, predictions):
        """
        计算预测置信度（基于树的预测一致性）
        :param predictions: 所有树的预测结果
        :return: 0-1之间的置信度
        """
        if len(predictions) <= 1:
            return 0.95  # 如果只有一个预测，返回默认置信度
        
        mean = sum(predictions) / len(predictions)
        
        # 计算标准差
        variance = sum((p - mean) ** 2 for p in predictions) / len(predictions)
        std_dev = np.sqrt(variance)
        
        # 计算变异系数（标准差/平均值），变异系数越小，置信度越高
        cv = std_dev / abs(mean) if mean != 0 else 1
        
        # 将变异系数转换为0~1之间的置信度，cv越小，置信度越高
        confidence = max(0, min(1, 1 - cv))
        
        return confidence

    def get_model_info(self):
        """
        获取模型信息
        :return: 模型信息
        """
        if not self.trained:
            return {
                'name': self.name,
                'trained': False,
                'features': []
            }
        
        # 确定树的数量
        num_trees = 0
        if self.model:
            num_trees = len(self.model.estimators_)
        elif self.custom_trees:
            num_trees = len(self.custom_trees)
        
        return {
            'name': self.name,
            'trained': True,
            'numTrees': num_trees,
            'features': self.feature_names
        }

    def train(self):
        """
        兼容旧的训练接口，直接加载预训练模型
        """
        self.load_model()
        return self 