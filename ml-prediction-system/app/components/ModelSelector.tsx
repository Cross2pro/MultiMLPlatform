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
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-bold text-gray-800">选择预测模型</h2>
        <span className="text-sm text-gray-500">{models.length} 个可用模型</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
        {models.map((model, index) => (
          <div 
            key={index}
            className={`p-3 border rounded-lg cursor-pointer transition-all duration-200 hover:shadow-md ${
              selectedModel === (index === 0 ? 'randomForest' : 'optimalNN')
                ? 'bg-blue-50 border-blue-500 shadow-md'
                : 'border-gray-200 hover:border-blue-300'
            }`}
            onClick={() => onSelectModel(index === 0 ? 'randomForest' : 'optimalNN')}
          >
            <h3 className="font-semibold text-base text-gray-800 mb-2">{model.name}</h3>
            <div className="space-y-1 text-xs text-gray-600">
              <div className="flex justify-between">
                <span>准确率:</span>
                <span className="font-medium text-blue-600">{(model.accuracy * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span>预测时间:</span>
                <span className="font-medium text-green-600">{model.predictTime}ms</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default ModelSelector; 