'use client';

import { useState, useEffect } from 'react';
import ModelSelector from './components/ModelSelector';
import FeatureInput from './components/FeatureInput';
import PredictionResult from './components/PredictionResult';

const API_URL = 'http://localhost:5000';

type Model = {
  name: string;
  accuracy: number;
  predictTime: string;
};

type PredictionResult = {
  modelName: string;
  confidence: number;
  processTime: string;
  individual_predictions: number[];
  shear_capacity: number;
};

export default function Home() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>('randomForest');
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string>('');

  useEffect(() => {
    // 获取可用模型列表
    async function fetchModels() {
      try {
        const response = await fetch(`${API_URL}/api/models`);
        if (!response.ok) {
          throw new Error('获取模型列表失败');
        }
        const data = await response.json();
        setModels(data);
      } catch (err) {
        console.error('获取模型时出错:', err);
        setError('无法连接到服务器。请确保后端服务正在运行。');
      }
    }

    fetchModels();
  }, []);

  const handleSelectModel = (model: string) => {
    setSelectedModel(model);
  };

  const handleSubmitFeatures = async (features: number[]) => {
    setLoading(true);
    setError('');
    
    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          modelType: selectedModel,
          features: features,
        }),
      });
      
      if (!response.ok) {
        throw new Error('预测请求失败');
      }
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error('预测过程中出错:', err);
      setError('预测失败。请稍后再试。');
    } finally {
      setLoading(false);
    }
  };

  // 用于测试UI展示的示例数据
  const handleDemoResult = () => {
    setResult({
      modelName: "随机森林",
      confidence: 0.8008928720291373,
      processTime: "0ms",
      individual_predictions: [383.7, 595.7, 620.1],
      shear_capacity: 533.1666666666666
    });
  };

  return (
    <main className="min-h-screen p-4 md:p-8 bg-gray-50">
      <div className="max-w-5xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-2">
            多模型机器学习预测系统
          </h1>
          <p className="text-gray-600">
            选择模型、输入特征数据，获取智能预测结果
          </p>
        </header>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            <p>{error}</p>
          </div>
        )}
        
        {models.length > 0 ? (
          <>
            <ModelSelector 
              models={models} 
              selectedModel={selectedModel}
              onSelectModel={handleSelectModel}
            />
            
            <FeatureInput 
              onSubmit={handleSubmitFeatures}
              isLoading={loading}
            />
            
            {result && <PredictionResult result={result} />}
            
            {/* 开发环境使用，方便测试UI */}
            {process.env.NODE_ENV === 'development' && !result && (
              <div className="mt-4">
                <button
                  onClick={handleDemoResult}
                  className="text-sm text-gray-500 underline"
                >
                  显示示例结果（仅开发环境）
                </button>
              </div>
            )}
          </>
        ) : !error ? (
          <div className="text-center py-8">
            <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent"></div>
            <p className="mt-4 text-gray-600">加载模型中...</p>
          </div>
        ) : null}
      </div>
      
      <footer className="text-center mt-12 py-4 text-sm text-gray-500 border-t">
        <p>© {new Date().getFullYear()} 长沙理工大学 - 李嘉俊</p>
      </footer>
    </main>
  );
}
