'use client';

import React from 'react';

type PredictionResultProps = {
  result: {
    modelName: string;
    confidence: number;
    processTime: string;
    individual_predictions: number[];
    shear_capacity: number;
  } | null;
};

const PredictionResult: React.FC<PredictionResultProps> = ({ result }) => {
  if (!result) return null;

  const { modelName, confidence, processTime, individual_predictions, shear_capacity } = result;
  
  return (
    <div className="w-full bg-white rounded-lg shadow-md border border-gray-200">
      {/* 标题 */}
      <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-4 rounded-t-lg">
        <h2 className="text-lg font-bold flex items-center">
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
          </svg>
          预测结果
        </h2>
      </div>
      
      <div className="p-4 space-y-4">
        {/* 主要预测结果 */}
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <div className="text-center">
            <div className="text-sm text-green-600 font-medium mb-1">剪切承载力</div>
            <div className="text-2xl font-bold text-green-700">{shear_capacity.toFixed(2)}</div>
            <div className="text-sm text-green-600">kN</div>
          </div>
        </div>
        
        {/* 模型信息 */}
        <div className="bg-blue-50 p-3 rounded-lg border border-blue-200">
          <h3 className="text-sm font-semibold text-blue-800 mb-2 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            模型信息
          </h3>
          
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">使用模型:</span>
              <span className="font-medium text-blue-800">{modelName}</span>
            </div>
            
            <div className="flex justify-between items-center">
              <span className="text-gray-600">置信度:</span>
              <div className="flex items-center">
                <div className="w-16 bg-gray-200 rounded-full h-1.5 mr-2">
                  <div 
                    className="bg-blue-600 h-1.5 rounded-full" 
                    style={{ width: `${confidence * 100}%` }}
                  ></div>
                </div>
                <span className="font-medium text-gray-700 text-xs">{(confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
            
            <div className="flex justify-between">
              <span className="text-gray-600">处理时间:</span>
              <span className="font-medium text-gray-700">{processTime}</span>
            </div>
          </div>
        </div>
        
        {/* 详细预测数据 */}
        <div className="bg-gray-50 p-3 rounded-lg border border-gray-200">
          <h3 className="text-sm font-semibold text-gray-800 mb-2 flex items-center">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            个体预测详情
          </h3>
          
          <div className="space-y-2">
            {individual_predictions.map((prediction, index) => {
              const deviation = prediction - shear_capacity;
              const deviationPercent = (deviation / shear_capacity) * 100;
              
              return (
                <div key={index} className="flex justify-between items-center py-1.5 px-2 bg-white rounded border text-sm">
                  <span className="text-gray-600">预测 {index + 1}:</span>
                  <div className="flex items-center space-x-2">
                    <span className="font-medium text-gray-800">{prediction.toFixed(1)}</span>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${
                      deviationPercent > 0 ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                    }`}>
                      {deviationPercent > 0 ? '+' : ''}{deviationPercent.toFixed(1)}%
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
        
        {/* 说明信息 */}
        <div className="bg-yellow-50 p-3 rounded-lg border border-yellow-200">
          <div className="flex items-start">
            <svg className="h-4 w-4 text-yellow-500 mt-0.5 mr-2 flex-shrink-0" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <div>
              <h3 className="text-sm font-medium text-yellow-800 mb-1">
                预测说明
              </h3>
              <p className="text-xs text-yellow-700 leading-relaxed">
                此预测结果仅供参考。实际剪切承载力可能受多种因素影响。最终结果由多次预测的平均值得出。
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult; 