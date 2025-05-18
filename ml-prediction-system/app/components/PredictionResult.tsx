'use client';

import React from 'react';

type PredictionResultProps = {
  result: {
    prediction: number;
    modelUsed: string;
    confidence: number;
    processTime: string;
  } | null;
};

const PredictionResult: React.FC<PredictionResultProps> = ({ result }) => {
  if (!result) return null;

  const { prediction, modelUsed, confidence, processTime } = result;
  
  // 对于随机森林模型，预测结果是0或1(分类)
  // 对于神经网络模型，预测结果是0-1之间的数值(回归)
  const isClassification = prediction === 0 || prediction === 1;
  
  return (
    <div className="w-full p-4 bg-white rounded-lg shadow-md mt-4 border-t-4 border-green-500">
      <h2 className="text-xl font-bold mb-4 text-gray-800">预测结果</h2>
      
      <div className="mb-4">
        <div className="flex items-center justify-between py-2 border-b">
          <span className="text-gray-700">使用模型:</span>
          <span className="font-semibold">{modelUsed}</span>
        </div>
        
        <div className="flex items-center justify-between py-2 border-b">
          <span className="text-gray-700">预测结果:</span>
          {isClassification ? (
            <span className={`font-bold ${prediction === 1 ? 'text-green-600' : 'text-red-600'}`}>
              {prediction === 1 ? '正类 (1)' : '负类 (0)'}
            </span>
          ) : (
            <span className="font-bold text-blue-600">{prediction.toFixed(4)}</span>
          )}
        </div>
        
        <div className="flex items-center justify-between py-2 border-b">
          <span className="text-gray-700">置信度:</span>
          <div className="flex items-center">
            <div className="w-32 bg-gray-200 rounded-full h-2.5 mr-2">
              <div 
                className="bg-blue-600 h-2.5 rounded-full" 
                style={{ width: `${confidence * 100}%` }}
              ></div>
            </div>
            <span className="font-semibold">{(confidence * 100).toFixed(1)}%</span>
          </div>
        </div>
        
        <div className="flex items-center justify-between py-2">
          <span className="text-gray-700">处理时间:</span>
          <span className="font-semibold">{processTime}</span>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult; 