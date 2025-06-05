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
    <div className="w-full p-6 bg-white rounded-lg shadow-md mt-6 border-t-4 border-blue-500">
      <h2 className="text-2xl font-bold mb-6 text-gray-800 flex items-center">
        <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-2 text-blue-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
        </svg>
        Prediction Result
      </h2>
      
      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-blue-50 p-4 rounded-lg">
          <div className="flex items-center mb-4">
            <div className="rounded-full bg-blue-100 p-2 mr-3">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-800">Model Information</h3>
          </div>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-blue-100">
              <span className="text-gray-700">Model Used:</span>
              <span className="font-semibold text-blue-800">{modelName}</span>
            </div>
            
            <div className="flex items-center justify-between py-2 border-b border-blue-100">
              <span className="text-gray-700">Confidence:</span>
              <div className="flex items-center">
                <div className="w-32 bg-gray-200 rounded-full h-2.5 mr-2">
                  <div 
                    className="bg-blue-600 h-2.5 rounded-full" 
                    style={{ width: `${confidence * 100}%` }}
                  ></div>
                </div>
                <span className="font-semibold text-gray-700">{(confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between py-2">
              <span className="text-gray-700">Processing Time:</span>
              <span className="font-semibold text-gray-700">{processTime}</span>
            </div>
          </div>
        </div>
        
        <div className="bg-green-50 p-4 rounded-lg">
          <div className="flex items-center mb-4">
            <div className="rounded-full bg-green-100 p-2 mr-3">
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-green-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-lg font-semibold text-gray-800">Prediction Result</h3>
          </div>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between py-2 border-b border-green-100">
              <span className="text-gray-700">Shear Capacity:</span>
              <span className="font-bold text-green-700">{shear_capacity.toFixed(2)} kN</span>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-6 bg-gray-50 p-4 rounded-lg">
        <div className="flex items-center mb-4">
          <div className="rounded-full bg-gray-200 p-2 mr-3">
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
          </div>
          <h3 className="text-lg font-semibold text-gray-800">Individual Prediction Details</h3>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-100">
              <tr>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Prediction No.
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Predicted Value (kN)
                </th>
                <th scope="col" className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Deviation from Average
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {individual_predictions.map((prediction, index) => {
                const deviation = prediction - shear_capacity;
                const deviationPercent = (deviation / shear_capacity) * 100;
                
                return (
                  <tr key={index}>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      Prediction {index + 1}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {prediction.toFixed(2)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                        deviationPercent > 0 ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                      }`}>
                        {deviationPercent > 0 ? '+' : ''}{deviationPercent.toFixed(2)}%
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
      
      <div className="mt-6 p-4 bg-yellow-50 rounded-lg border border-yellow-100">
        <div className="flex items-start">
          <div className="flex-shrink-0">
            <svg className="h-5 w-5 text-yellow-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
          </div>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-yellow-800">
              Model Description
            </h3>
            <div className="mt-2 text-sm text-yellow-700">
              <p>
                This prediction result is for reference only. The actual shear capacity may be affected by various factors. The final result is derived from the average of multiple predictions.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionResult; 