import numpy as np

class ModelUtils:
    @staticmethod
    def split_dataset(dataset, test_ratio=0.2):
        """
        将数据集划分为训练集和测试集
        :param dataset: 完整数据集
        :param test_ratio: 测试集比例，默认为0.2
        :return: 包含训练集和测试集的对象
        """
        # 打乱数据集
        np.random.shuffle(dataset)
        
        test_size = int(len(dataset) * test_ratio)
        test_set = dataset[:test_size]
        train_set = dataset[test_size:]
        
        return {'trainSet': train_set, 'testSet': test_set}

    @staticmethod
    def calculate_mse(actual, predicted):
        """
        计算预测结果的均方误差
        :param actual: 实际值数组
        :param predicted: 预测值数组
        :return: 均方误差
        """
        if len(actual) != len(predicted) or len(actual) == 0:
            raise ValueError('输入数组长度不匹配或为空')
        
        return np.mean((np.array(actual) - np.array(predicted)) ** 2)

    @staticmethod
    def calculate_rmse(actual, predicted):
        """
        计算预测结果的均方根误差
        :param actual: 实际值数组
        :param predicted: 预测值数组
        :return: 均方根误差
        """
        return np.sqrt(ModelUtils.calculate_mse(actual, predicted))

    @staticmethod
    def calculate_r2(actual, predicted):
        """
        计算预测结果的决定系数 R²
        :param actual: 实际值数组
        :param predicted: 预测值数组
        :return: 决定系数 R²
        """
        if len(actual) != len(predicted) or len(actual) == 0:
            raise ValueError('输入数组长度不匹配或为空')
        
        actual_array = np.array(actual)
        predicted_array = np.array(predicted)
        
        mean = np.mean(actual_array)
        total_variation = np.sum((actual_array - mean) ** 2)
        residual_variation = np.sum((actual_array - predicted_array) ** 2)
        
        return 1 - (residual_variation / total_variation)

    @staticmethod
    def calculate_mae(actual, predicted):
        """
        计算预测结果的平均绝对误差
        :param actual: 实际值数组
        :param predicted: 预测值数组
        :return: 平均绝对误差
        """
        if len(actual) != len(predicted) or len(actual) == 0:
            raise ValueError('输入数组长度不匹配或为空')
        
        return np.mean(np.abs(np.array(actual) - np.array(predicted)))
    
    @staticmethod
    def evaluate_model(actual, predicted):
        """
        计算模型性能指标
        :param actual: 实际值数组
        :param predicted: 预测值数组
        :return: 包含各项性能指标的对象
        """
        return {
            'mse': ModelUtils.calculate_mse(actual, predicted),
            'rmse': ModelUtils.calculate_rmse(actual, predicted),
            'mae': ModelUtils.calculate_mae(actual, predicted),
            'r2': ModelUtils.calculate_r2(actual, predicted)
        }
    
    @staticmethod
    def standardize_data(data):
        """
        标准化特征数据
        :param data: 要标准化的数据数组
        :return: 包含标准化后的数据及标准化参数
        """
        data_array = np.array(data)
        mean = np.mean(data_array)
        std_dev = np.std(data_array)
        
        standardized = (data_array - mean) / (std_dev if std_dev != 0 else 1)
        
        return {
            'standardizedData': standardized.tolist(),
            'mean': mean,
            'stdDev': std_dev
        }
    
    @staticmethod
    def standardize_value(value, mean, std_dev):
        """
        使用已知参数标准化单个特征值
        :param value: 特征值
        :param mean: 均值
        :param std_dev: 标准差
        :return: 标准化后的特征值
        """
        return (value - mean) / (std_dev if std_dev != 0 else 1)
    
    @staticmethod
    def unstandardize_value(standardized_value, mean, std_dev):
        """
        将标准化后的值转换回原始值
        :param standardized_value: 标准化后的值
        :param mean: 均值
        :param std_dev: 标准差
        :return: 原始值
        """
        return standardized_value * (std_dev if std_dev != 0 else 1) + mean 