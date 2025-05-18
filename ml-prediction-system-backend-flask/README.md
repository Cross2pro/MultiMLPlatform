# UHPC接缝抗剪承载力预测系统后端 (Flask版)

这个项目是基于机器学习的UHPC接缝抗剪承载力预测系统的后端部分，使用Python和Flask实现。

## 功能特点

- 基于随机森林算法的预测模型
- RESTful API接口支持模型训练、预测和评估
- 支持多种预测特征参数
- 可扩展的模型架构，方便添加新模型

## 技术栈

- Python 3.8+
- Flask
- NumPy
- Scikit-learn

## 预测特征参数

### 配置类参数
- Jt (joint_type): 接缝类型，1=干接缝，2=环氧树脂接缝，3=湿接缝，4=整体浇筑接缝
- St (specimen_type): 试件类型，1=单接缝(SJ)，2=双接缝(DJ)
- Nk (key_number): 键槽数量

### 几何尺寸参数
- Bk (key_width): 单个键槽宽度
- Hk (key_root_height): 单个键槽根部高度
- Dk (key_depth): 单个键槽深度
- theta_k (key_inclination): 单个键槽倾角
- Sk (key_spacing): 相邻键槽间距
- hk (key_front_height): 单个键槽前部高度
- Dk/Hk (key_depth_height_ratio): 键槽深度与高度比
- Bj (joint_width): 接缝宽度
- Hj (joint_height): 接缝高度
- Ak (key_area): 接缝中键槽区域面积
- Aj (joint_area): 接缝总面积
- Asm (flat_region_area): 接缝中平坦区域面积
- Ak/Aj (key_joint_area_ratio): 键槽面积与接缝面积比率

### UHPC材料性能参数
- fc (compressive_strength): 抗压强度
- Ft (fiber_type): 纤维类型，0=无纤维，1=直纤维，2=混合纤维(直纤维和端钩纤维)，3=端钩纤维
- pf (fiber_volume_fraction): 纤维体积分数
- lf (fiber_length): 纤维长度
- df (fiber_diameter): 纤维直径
- lambda_f (fiber_reinforcing_index): 纤维增强指数，计算公式为 pf×lf/df

### 约束应力参数
- sigma_n (confining_stress): 约束应力
- sigma_n/fc (confining_ratio): 约束比，约束应力与抗压强度之比

### 输出变量
- Vu (shear_capacity): 抗剪承载力，作为模型的预测目标

## 项目结构

```
ml-prediction-system-backend-flask/
├── app/
│   ├── data/
│   │   └── rf_model_simplified.json  # 预训练模型数据
│   ├── models/
│   │   ├── random_forest.py          # 随机森林模型实现
│   │   └── model_manager.py          # 模型管理器
│   ├── routes/
│   │   └── model_routes.py           # API路由
│   ├── utils/
│   │   └── model_utils.py            # 工具函数
│   ├── static/                       # 静态文件
│   ├── templates/                    # 模板文件
│   └── __init__.py                   # Flask应用初始化
├── run.py                            # 应用入口
├── requirements.txt                  # 项目依赖
└── README.md                         # 项目说明
```

## API接口

### 获取所有可用模型
```
GET /api/models
```

### 加载模型
```
POST /api/load
Content-Type: application/json

{
  "modelType": "randomForest",
  "modelPath": "path/to/model.json" // 可选
}
```

### 预测
```
POST /api/predict
Content-Type: application/json

{
  "modelType": "randomForest",
  "features": {
    "joint_type": 2,
    "specimen_type": 1,
    "key_number": 3,
    "key_width": 50,
    "key_root_height": 20,
    "key_depth": 15,
    "key_inclination": 45,
    "key_spacing": 100,
    "key_front_height": 25,
    "key_depth_height_ratio": 0.75,
    "joint_width": 200,
    "joint_height": 150,
    "key_area": 1500,
    "joint_area": 30000,
    "flat_region_area": 28500,
    "key_joint_area_ratio": 0.05,
    "compressive_strength": 120,
    "fiber_type": 3,
    "fiber_volume_fraction": 0.02,
    "fiber_length": 13,
    "fiber_diameter": 0.2,
    "fiber_reinforcing_index": 1.3,
    "confining_stress": 5,
    "confining_ratio": 0.042
  }
}
```

### 检查模型状态
```
GET /api/status/<model_type>
```

## 安装和运行

1. 安装依赖
```bash
pip install -r requirements.txt
```

2. 启动服务器
```bash
python run.py
```

默认服务器运行在 http://localhost:5000 

## 模型支持

系统目前支持以下预测模型:

- **随机森林模型（Random Forest）**: 基于多棵决策树的集成学习模型，用于预测剪切承载力。
- **最优神经网络模型（OptimalNN）**: 基于PyTorch的优化神经网络模型，提供高精度预测。

### 随机森林模型

随机森林模型使用多棵决策树的集成来预测UHPC接缝的抗剪承载力。模型基于JSON格式存储，支持直接的预测和置信度评估。

### OptimalNN模型

OptimalNN模型是基于PyTorch实现的神经网络模型，采用多层感知机（MLP）结构，用于预测UHPC接缝的抗剪承载力。

#### 模型架构

OptimalNN采用以下网络结构：
- 输入层：24个特征节点
- 隐藏层1：64个节点，ReLU激活函数，BatchNorm和Dropout(0.2)
- 隐藏层2：32个节点，ReLU激活函数，BatchNorm和Dropout(0.2)
- 输出层：1个节点（抗剪承载力预测值）

#### 模型文件

默认模型文件位置：`app/data/OptimalNN_model.pt`

模型文件格式为PyTorch的.pt格式，包含以下信息：
- 模型状态字典（权重和偏置）
- 输入维度和隐藏层维度
- 特征名称列表
- 特征缩放器（可选）

#### 特征需求

OptimalNN模型支持与随机森林模型相同的24个特征输入，包括接缝几何参数、材料属性和应力条件等。

#### 容错机制

OptimalNN模型具有强大的容错机制：
1. 输入特征验证和范围检查
2. 预测结果合理性验证
3. 当模型不可用或预测结果不合理时，会自动使用基于专业知识的估算方法
4. 支持模拟多样本预测，提供类似集成模型的置信度评估

#### API使用示例

```python
# 加载模型
POST /api/model/load
{
    "modelType": "optimalNN"
}

# 使用模型预测
POST /api/model/predict
{
    "modelType": "optimalNN",
    "features": {
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
}
```

#### 预测结果格式

```json
{
    "shear_capacity": 450.8,
    "individual_predictions": [467.2, 413.5, 499.0, 438.2, 488.8],
    "confidence": 0.93,
    "modelName": "最优神经网络",
    "processTime": "2ms"
}
```

### 模型对比

两种模型各有优势：
- 随机森林模型：对非线性关系建模能力强，适合有限数据集
- OptimalNN模型：泛化能力强，适合大数据集和复杂模式识别 