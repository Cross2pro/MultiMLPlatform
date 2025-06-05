from flask import Flask, send_from_directory, send_file
from flask_cors import CORS
import os

def create_app():
    app = Flask(__name__, static_folder='static', static_url_path='/static')
    CORS(app)  # 启用跨域请求支持
    
    # 注册蓝图
    from app.routes.model_routes import model_bp
    app.register_blueprint(model_bp, url_prefix='/api')
    
    # 前端静态文件服务
    @app.route('/')
    def serve_frontend():
        """服务前端首页"""
        frontend_path = os.path.join(app.static_folder, 'frontend')
        if os.path.exists(os.path.join(frontend_path, 'index.html')):
            return send_file(os.path.join(frontend_path, 'index.html'))
        else:
            # 如果前端文件不存在，返回API信息
            return app_info()
    
    @app.route('/<path:path>')
    def serve_frontend_routes(path):
        """服务前端路由和静态资源"""
        frontend_path = os.path.join(app.static_folder, 'frontend')
        
        # 首先尝试找到具体文件
        file_path = os.path.join(frontend_path, path)
        if os.path.isfile(file_path):
            return send_file(file_path)
        
        # 如果是目录且包含 index.html，返回 index.html
        if os.path.isdir(file_path):
            index_path = os.path.join(file_path, 'index.html')
            if os.path.exists(index_path):
                return send_file(index_path)
        
        # 对于 SPA 路由，返回主 index.html
        main_index = os.path.join(frontend_path, 'index.html')
        if os.path.exists(main_index):
            return send_file(main_index)
        
        # 如果前端文件不存在，返回 404
        return {'error': 'Not found'}, 404
    
    # 健康检查接口
    @app.route('/health')
    def health_check():
        from datetime import datetime
        return {'status': 'ok', 'timestamp': datetime.now().isoformat()}
    
    # API 信息接口（当前端文件不存在时显示）
    def app_info():
        return {
            'name': 'UHPC接缝抗剪承载力预测系统',
            'version': '1.0.0',
            'description': '基于预训练随机森林模型的UHPC接缝抗剪承载力预测系统后端',
            'frontend_status': '前端文件已集成' if os.path.exists(os.path.join(app.static_folder, 'frontend', 'index.html')) else '前端文件未找到',
            'endpoints': [
                {'path': '/api/models', 'method': 'GET', 'description': '获取所有可用模型'},
                {'path': '/api/load', 'method': 'POST', 'description': '加载预训练模型'},
                {'path': '/api/predict', 'method': 'POST', 'description': '使用模型进行预测'},
                {'path': '/api/status/<model_type>', 'method': 'GET', 'description': '检查模型状态'},
                {'path': '/health', 'method': 'GET', 'description': '健康检查'}
            ]
        }
    
    return app 