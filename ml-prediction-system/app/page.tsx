'use client';

import { useState, useEffect } from 'react';
import ModelSelector from './components/ModelSelector';
import FeatureInput from './components/FeatureInput';
import PredictionResult from './components/PredictionResult';

// 使用相对路径，这样前后端集成后就会使用同一个域名和端口
const API_URL = process.env.NODE_ENV === 'development' ? 'http://localhost:5000' : '';

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
    <main className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto p-4 md:p-6">
        <header className="text-center mb-6">
          <h1 className="text-3xl md:text-4xl font-bold text-gray-800 mb-2">
            多模型机器学习预测系统
          </h1>
          <p className="text-gray-600">
          © {new Date().getFullYear()} 长沙理工大学 - 李嘉俊
          </p>
        </header>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-6">
            <p>{error}</p>
          </div>
        )}
        
        {models.length > 0 ? (
          <>
            {/* 模型选择器保持在顶部 */}
            <div className="mb-6">
              <ModelSelector 
                models={models} 
                selectedModel={selectedModel}
                onSelectModel={handleSelectModel}
              />
            </div>
            
            {/* 主要内容区域：左侧参数输入，右侧预测结果 */}
            <div className="grid lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {/* 左侧参数输入区域 */}
              <div className="lg:col-span-2 xl:col-span-3">
                <FeatureInput 
                  onSubmit={handleSubmitFeatures}
                  isLoading={loading}
                />
              </div>
              
              {/* 右侧预测结果区域 */}
              <div className="lg:col-span-1 xl:col-span-1">
                <div className="sticky top-6">
                  {result ? (
                    <PredictionResult result={result} />
                  ) : (
                    <div className="bg-white p-6 rounded-lg shadow-md border-2 border-dashed border-gray-300">
                      <div className="text-center text-gray-500">
                        <svg className="mx-auto h-12 w-12 text-gray-400 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                        </svg>
                        <h3 className="text-lg font-medium text-gray-800 mb-2">等待预测结果</h3>
                        <p className="text-sm text-gray-600">
                          请在左侧输入参数并点击&ldquo;开始预测&rdquo;按钮
                        </p>
                        {process.env.NODE_ENV === 'development' && (
                          <button
                            onClick={handleDemoResult}
                            className="mt-4 text-sm text-blue-600 underline hover:text-blue-800"
                          >
                            显示示例结果（仅开发环境）
                          </button>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </>
        ) : !error ? (
          <div className="text-center py-16">
            <div className="inline-block h-12 w-12 animate-spin rounded-full border-4 border-solid border-blue-600 border-r-transparent mb-4"></div>
            <h3 className="text-lg font-medium text-gray-800 mb-2">正在加载模型</h3>
            <p className="text-gray-600">请稍等，正在连接服务器...</p>
          </div>
        ) : null}
      </div>
      
      <footer className="text-center mt-12 py-4 text-sm text-gray-500 border-t">
        <p>© {new Date().getFullYear()} 长沙理工大学 - 李嘉俊</p>
        <a href="https://beian.miit.gov.cn/" target="_blank" rel="noopener noreferrer">湘ICP备2021017124号-1</a>
      </footer>
    </main>
  );
}
