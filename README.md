# 多模型机器学习预测系统

这是一个使用Next.js和Express.js构建的机器学习预测系统，支持多种预测模型，包括随机森林和人工神经网络。

## 项目结构

该项目分为两个主要部分：

1. **前端**（`ml-prediction-system`）：使用Next.js和Tailwind CSS构建
2. **后端**（`ml-prediction-system-backend`）：使用Express.js构建

## 功能特点

- 支持多种机器学习模型（随机森林和人工神经网络）
- 可配置特征数量
- 实时预测结果展示
- 现代化UI设计，响应式布局

## 快速开始

### 启动后端服务

```bash
cd ml-prediction-system-backend
npm install
npm run dev
```

后端服务将在 http://localhost:5000 上运行。

### 启动前端应用

```bash
cd ml-prediction-system
npm install
npm run dev
```

前端应用将在 http://localhost:3000 上运行。

## 技术栈

### 前端
- Next.js
- TypeScript
- Tailwind CSS
- React Hooks

### 后端
- Express.js
- Node.js
- REST API

## 项目扩展

要添加更多机器学习模型，您可以在后端的`index.js`文件中扩展`models`对象，并实现相应的预测逻辑。 