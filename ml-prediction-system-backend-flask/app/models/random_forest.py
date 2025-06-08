import os
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

class RandomForestModel:
    def __init__(self):
        self.name = '随机森林'
        self.trained = False
        self.model = None
        self.pipeline = None
        self.feature_names = None
        self.scaler = None
        self.categorical_info = None

    def load_model(self, model_path=None, pipeline_path=None):
        """
        加载预训练模型和预处理管道
        :param model_path: 模型路径，不提供则使用默认路径
        :param pipeline_path: 预处理管道路径，不提供则使用默认路径
        :return: 是否加载成功
        """
        try:
            # 确定模型文件路径
            if model_path:
                model_file_path = model_path
            else:
                model_file_path = os.path.join(os.path.dirname(__file__), '../data/RandomForest_model.pkl')
            
            # 确定预处理管道路径
            if pipeline_path:
                pipeline_file_path = pipeline_path
            else:
                pipeline_file_path = os.path.join(os.path.dirname(__file__), '../data/preprocessing_pipeline.pkl')
            
            # 检查文件是否存在
            if not os.path.exists(model_file_path):
                raise FileNotFoundError(f"模型文件不存在: {model_file_path}")
            
            if not os.path.exists(pipeline_file_path):
                raise FileNotFoundError(f"预处理管道文件不存在: {pipeline_file_path}")
            
            # 加载模型
            with open(model_file_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # 加载预处理管道
            with open(pipeline_file_path, 'rb') as f:
                self.pipeline = pickle.load(f)
            
            # 从管道中提取信息
            self.feature_names = self.pipeline['feature_names']
            self.scaler = self.pipeline['scaler']
            self.categorical_info = self.pipeline.get('categorical_info', {})
            
            self.trained = True
            print('随机森林模型和预处理管道加载成功!')
            return True
            
        except Exception as e:
            print(f'加载模型失败: {str(e)}')
            raise Exception(f"加载模型失败: {str(e)}")

    def preprocess_new_data(self, new_data):
        """
        预处理新数据
        :param new_data: 输入数据（字典格式）
        :return: 预处理后的数据
        """
        # 将字典转换为DataFrame
        if isinstance(new_data, dict):
            # 如果是单个样本的字典，转换为DataFrame
            df = pd.DataFrame([new_data])
        else:
            df = new_data.copy()
        
        # 移除非特征列（如果存在）
        drop_cols = ['id', 'reference', 'specimen']
        for col in drop_cols:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # 独热编码分类变量
        categorical_cols = list(self.categorical_info.keys())
        if categorical_cols:
            # 只对存在的分类列进行独热编码
            existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
            if existing_categorical_cols:
                df = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True)
        
        # 确保所有需要的特征都存在
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            # 对于缺失的特征，添加默认值0
            for col in missing_cols:
                df[col] = 0
        
        # 仅保留模型需要的特征，按正确顺序排列
        df = df[self.feature_names]
        
        # 标准化
        df_scaled = self.scaler.transform(df)
        
        return df_scaled

    def predict(self, features, outlier_threshold=1.5):
        """
        使用预训练随机森林模型进行预测，并过滤误差较大的估计器
        :param features: 输入特征（字典格式）
        :param outlier_threshold: 过滤阈值（标准差的倍数），默认1.5
        :return: 预测结果
        """
        # 如果模型尚未加载，则加载模型
        if not self.trained:
            self.load_model()

        # 预处理特征数据
        try:
            X_processed = self.preprocess_new_data(features)
            
            # 获取各棵树的预测
            individual_predictions = []
            for tree in self.model.estimators_:
                tree_prediction = float(tree.predict(X_processed)[0])
                individual_predictions.append(tree_prediction)
            
            # 过滤误差较大的估计器
            filtered_predictions = self._filter_outlier_estimators(
                individual_predictions, outlier_threshold
            )
            
            # 使用过滤后的预测计算最终结果
            final_prediction = sum(filtered_predictions) / len(filtered_predictions)
            
            return {
                'shear_capacity': float(final_prediction),
                'individual_predictions': individual_predictions,
                'filtered_predictions': filtered_predictions,
                'used_estimators': len(filtered_predictions),
                'total_estimators': len(individual_predictions),
                'filtered_out': len(individual_predictions) - len(filtered_predictions),
                'confidence': self.calculate_confidence(filtered_predictions)
            }
            
        except Exception as e:
            raise Exception(f"预测过程中出现错误: {str(e)}")

    def _filter_outlier_estimators(self, predictions, threshold=1.5):
        """
        过滤误差较大的估计器
        :param predictions: 所有估计器的预测结果
        :param threshold: 过滤阈值（标准差的倍数）
        :return: 过滤后的预测结果
        """
        if len(predictions) <= 2:
            # 如果预测数量太少，不进行过滤
            return predictions
        
        # 计算平均值和标准差
        mean_pred = sum(predictions) / len(predictions)
        variance = sum((p - mean_pred) ** 2 for p in predictions) / len(predictions)
        std_dev = np.sqrt(variance)
        
        # 如果标准差太小，说明所有预测都很接近，不需要过滤
        if std_dev < 1e-6:
            return predictions
        
        # 过滤超出阈值的预测
        filtered_predictions = []
        for pred in predictions:
            # 计算Z分数（标准化后的偏差）
            z_score = abs(pred - mean_pred) / std_dev
            if z_score <= threshold:
                filtered_predictions.append(pred)
        
        # 确保至少保留一半的估计器
        min_estimators = max(1, len(predictions) // 2)
        if len(filtered_predictions) < min_estimators:
            # 如果过滤后剩余太少，按距离平均值的远近重新选择
            predictions_with_distance = [
                (pred, abs(pred - mean_pred)) for pred in predictions
            ]
            # 按距离排序，选择距离最近的一半估计器
            predictions_with_distance.sort(key=lambda x: x[1])
            filtered_predictions = [
                pred for pred, _ in predictions_with_distance[:min_estimators]
            ]
        
        return filtered_predictions

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
        
        # 获取树的数量
        num_trees = len(self.model.estimators_) if self.model else 0
        
        return {
            'name': self.name,
            'trained': True,
            'numTrees': num_trees,
            'features': self.feature_names,
            'totalFeatures': len(self.feature_names) if self.feature_names else 0,
            'categoricalInfo': self.categorical_info
        }

    def train(self):
        """
        兼容旧的训练接口，直接加载预训练模型
        """
        self.load_model()
        return self 