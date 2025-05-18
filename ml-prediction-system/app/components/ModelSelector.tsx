'use client';

import React from 'react';

type Model = {
  name: string;
  accuracy: number;
  predictTime: string;
};

type ModelSelectorProps = {
  models: Model[];
  selectedModel: string;
  onSelectModel: (model: string) => void;
};

const ModelSelector: React.FC<ModelSelectorProps> = ({
  models,
  selectedModel,
  onSelectModel
}) => {
  return (
    <div className="w-full p-4 bg-white rounded-lg shadow-md">
      <h2 className="text-xl font-bold mb-4 text-gray-800">选择预测模型</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {models.map((model, index) => (
          <div 
            key={index}
            className={`p-4 border rounded-lg cursor-pointer transition-all duration-300 ${
              selectedModel === (index === 0 ? 'randomForest' : 'optimalNN')
                ? 'bg-blue-50 border-blue-500 shadow-md'
                : 'border-gray-200 hover:border-blue-300'
            }`}
            onClick={() => onSelectModel(index === 0 ? 'randomForest' : 'optimalNN')}
          >
            <h3 className="font-semibold text-lg text-gray-800">{model.name}</h3>
            <div className="mt-2 text-sm text-gray-600">
              <p>准确率: <span className="font-medium">{(model.accuracy * 100).toFixed(1)}%</span></p>
              <p>平均预测时间: <span className="font-medium">{model.predictTime}ms</span></p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelSelector; 