from flask import Flask
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)  # 启用跨域请求支持
    
    # 注册蓝图
    from app.routes.model_routes import model_bp
    app.register_blueprint(model_bp, url_prefix='/api')
    
    # 健康检查接口
    @app.route('/health')
    def health_check():
        from datetime import datetime
        return {'status': 'ok', 'timestamp': datetime.now().isoformat()}
    
    # 应用信息接口
    @app.route('/')
    def app_info():
        return {
            'name': 'UHPC接缝抗剪承载力预测系统',
            'version': '1.0.0',
            'description': '基于预训练随机森林模型的UHPC接缝抗剪承载力预测系统后端',
            'endpoints': [
                {'path': '/api/models', 'method': 'GET', 'description': '获取所有可用模型'},
                {'path': '/api/load', 'method': 'POST', 'description': '加载预训练模型'},
                {'path': '/api/predict', 'method': 'POST', 'description': '使用模型进行预测'},
                {'path': '/api/status/<model_type>', 'method': 'GET', 'description': '检查模型状态'},
                {'path': '/health', 'method': 'GET', 'description': '健康检查'}
            ]
        }
    
    return app 