from flask import Blueprint, request, jsonify
from app.models.model_manager import model_manager

model_bp = Blueprint('model', __name__)

@model_bp.route('/models', methods=['GET'])
def get_models():
    """获取所有可用模型信息"""
    try:
        models = model_manager.get_available_models()
        return jsonify(models)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_bp.route('/load', methods=['POST'])
def load_model():
    """加载预训练模型"""
    try:
        data = request.get_json()
        model_type = data.get('modelType')
        model_path = data.get('modelPath')
        
        if not model_type:
            return jsonify({'error': '请提供模型类型'}), 400
        
        result = model_manager.load_model(model_type, model_path)
        return jsonify({'success': True, 'message': '模型加载成功', 'model': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_bp.route('/predict', methods=['POST'])
def predict():
    """使用指定模型进行预测"""
    try:
        data = request.get_json()
        model_type = data.get('modelType')
        features = data.get('features')
        
        if not model_type or not features:
            return jsonify({'error': '请提供模型类型和特征数据'}), 400
        
        # 验证特征数据
        if isinstance(features, list):
            # 如果是数组，转换为字典格式 (第一版前端传来的是数组)
            features_dict = {}
            # 获取当前模型对象
            model_obj = model_manager.models.get(model_type)
            if not model_obj:
                return jsonify({'error': f'模型类型不存在: {model_type}'}), 400
            # 使用模型的特征名称列表来构建字典
            for i, feature_name in enumerate(model_obj.feature_names):
                if i < len(features):
                    features_dict[feature_name] = features[i]
                else:
                    features_dict[feature_name] = 0
            features = features_dict
        elif not isinstance(features, dict):
            return jsonify({'error': '特征数据格式不正确'}), 400
        
        result = model_manager.predict(model_type, features)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@model_bp.route('/status/<model_type>', methods=['GET'])
def get_model_status(model_type):
    """检查模型状态"""
    try:
        models = model_manager.get_available_models()
        model = next((m for m in models if m['id'] == model_type), None)
        
        if not model:
            return jsonify({'error': '模型不存在'}), 404
        
        return jsonify({
            'modelType': model_type,
            'name': model['name'],
            'trained': model['trained'],
            'status': '已加载' if model['trained'] else '未加载'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500 