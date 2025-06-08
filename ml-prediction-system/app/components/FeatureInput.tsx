'use client';

import React, { useState } from 'react';

type FeatureInputProps = {
  onSubmit: (features: number[]) => void;
  isLoading: boolean;
};

// 定义字段类型和初始值
type FieldConfig = {
  id: string;
  label: string;
  description: string;
  type: 'number' | 'select';
  options?: {value: number, label: string}[];
  defaultValue: number;
  step?: number;
  min?: number;
};

type FieldGroup = {
  title: string;
  fields: FieldConfig[];
  color: string; // 添加颜色主题
};

const FeatureInput: React.FC<FeatureInputProps> = ({ onSubmit, isLoading }) => {
  // 字段配置 - 添加颜色主题
  const fieldGroups: FieldGroup[] = [
    {
      title: "Configuration Parameters",
      color: "blue",
      fields: [
        {
          id: "joint_type",
          label: "Joint Type (Jt)",
          description: "1=干接缝，2=环氧树脂接缝，3=湿接缝，4=整体浇筑接缝",
          type: "select" as const,
          options: [
            { value: 1, label: "Dry Joint" },
            { value: 2, label: "Epoxy Joint" },
            { value: 3, label: "Wet Joint" },
            { value: 4, label: "Monolithic Joint" }
          ],
          defaultValue: 1
        },
        {
          id: "specimen_type",
          label: "Specimen Type (St)",
          description: "1=单接缝(SJ)，2=双接缝(DJ)",
          type: "select" as const,
          options: [
            { value: 1, label: "Single Joint (SJ)" },
            { value: 2, label: "Double Joint (DJ)" }
          ],
          defaultValue: 1
        },
        {
          id: "key_number",
          label: "Number of Keys (Nk)",
          description: "接缝中键槽的数量",
          type: "number" as const,
          defaultValue: 1,
          min: 0,
          step: 1
        }
      ]
    },
    {
      title: "Geometric Parameters",
      color: "green",
      fields: [
        {
          id: "key_width",
          label: "Key Width (Bk)",
          description: "单个键槽的宽度",
          type: "number" as const,
          defaultValue: 35,
          step: 0.1,
          min: 0
        },
        {
          id: "key_root_height",
          label: "Key Root Height (Hk)",
          description: "单个键槽根部的高度",
          type: "number" as const,
          defaultValue: 50,
          step: 0.1,
          min: 0
        },
        {
          id: "key_depth",
          label: "Key Depth (Dk)",
          description: "单个键槽的深度",
          type: "number" as const,
          defaultValue: 25,
          step: 0.1,
          min: 0
        },
        {
          id: "key_inclination",
          label: "Key Inclination (theta_k)",
          description: "单个键槽的倾斜角度",
          type: "number" as const,
          defaultValue: 34.5,
          step: 0.1
        },
        {
          id: "key_spacing",
          label: "Key Spacing (Sk)",
          description: "相邻键槽之间的距离",
          type: "number" as const,
          defaultValue: 50,
          step: 0.1,
          min: 0
        },
        {
          id: "key_front_height",
          label: "Key Front Height (hk)",
          description: "单个键槽前部的高度",
          type: "number" as const,
          defaultValue: 25,
          step: 0.1,
          min: 0
        },
        {
          id: "key_depth_height_ratio",
          label: "Key Depth-Height Ratio (Dk/Hk)",
          description: "键槽深度与高度的比值",
          type: "number" as const,
          defaultValue: 0.5,
          step: 0.01,
          min: 0
        },
        {
          id: "joint_width",
          label: "Joint Width (Bj)",
          description: "接缝的总宽度",
          type: "number" as const,
          defaultValue: 200,
          step: 0.1,
          min: 0
        },
        {
          id: "joint_height",
          label: "Joint Height (Hj)",
          description: "接缝的总高度",
          type: "number" as const,
          defaultValue: 200,
          step: 0.1,
          min: 0
        },
        {
          id: "key_area",
          label: "Key Area (Ak)",
          description: "接缝中键槽区域的总面积",
          type: "number" as const,
          defaultValue: 17500,
          step: 0.1,
          min: 0
        },
        {
          id: "joint_area",
          label: "Joint Area (Aj)",
          description: "接缝的总面积",
          type: "number" as const,
          defaultValue: 40000,
          step: 0.1,
          min: 0
        },
        {
          id: "flat_region_area",
          label: "Flat Region Area (Asm)",
          description: "接缝中平坦区域的面积",
          type: "number" as const,
          defaultValue: 38250,
          step: 0.1,
          min: 0
        },
        {
          id: "key_joint_area_ratio",
          label: "Key-Joint Area Ratio (Ak/Aj)",
          description: "键槽面积与接缝面积的比值",
          type: "number" as const,
          defaultValue: 0.04,
          step: 0.01,
          min: 0
        }
      ]
    },
    {
      title: "UHPC Material Properties",
      color: "purple",
      fields: [
        {
          id: "compressive_strength",
          label: "Compressive Strength (fc)",
          description: "UHPC材料的抗压强度",
          type: "number" as const,
          defaultValue: 193,
          step: 0.1,
          min: 0
        },
        {
          id: "fiber_type",
          label: "Fiber Type (Ft)",
          description: "0=无纤维，1=直纤维，2=混合纤维(直纤维和端钩纤维)，3=端钩纤维",
          type: "select" as const,
          options: [
            { value: 0, label: "No Fiber" },
            { value: 1, label: "Straight Fiber" },
            { value: 2, label: "Mixed Fiber (Straight and Hooked)" },
            { value: 3, label: "Hooked Fiber" }
          ],
          defaultValue: 1
        },
        {
          id: "fiber_volume_fraction",
          label: "Fiber Volume Fraction (pf)",
          description: "纤维在UHPC中的体积分数",
          type: "number" as const,
          defaultValue: 0.01,
          step: 0.01,
          min: 0
        },
        {
          id: "fiber_length",
          label: "Fiber Length (lf)",
          description: "纤维的长度",
          type: "number" as const,
          defaultValue: 13,
          step: 0.1,
          min: 0
        },
        {
          id: "fiber_diameter",
          label: "Fiber Diameter (df)",
          description: "纤维的直径",
          type: "number" as const,
          defaultValue: 0.2,
          step: 0.01,
          min: 0
        },
        {
          id: "fiber_reinforcing_index",
          label: "Fiber Reinforcing Index (lambda_f)",
          description: "纤维增强指数，计算公式为 pf×lf/df",
          type: "number" as const,
          defaultValue: 65,
          step: 0.01,
          min: 0
        }
      ]
    },
    {
      title: "Confinement Stress Parameters",
      color: "orange",
      fields: [
        {
          id: "confining_stress",
          label: "Confinement Stress (sigma_n)",
          description: "约束应力值",
          type: "number" as const,
          defaultValue: 1,
          step: 0.1,
          min: 0
        },
        {
          id: "confining_ratio",
          label: "Confinement Ratio (sigma_n/fc)",
          description: "约束应力与抗压强度之比",
          type: "number" as const,
          defaultValue: 0.005,
          step: 0.001,
          min: 0
        }
      ]
    },
    {
      title: "Shear Strength",
      color: "red",
      fields: [
        {
          id: "shear_strength",
          label: "Shear Strength (tau_max)",
          description: "接缝的最大抗剪强度",
          type: "number" as const,
          defaultValue: 1421,
          step: 0.1,
          min: 0
        }
      ]
    }
  ];

  // 初始化所有字段的值
  const initialValues: Record<string, number> = {};
  fieldGroups.forEach(group => {
    group.fields.forEach(field => {
      initialValues[field.id] = field.defaultValue;
    });
  });

  const [values, setValues] = useState<Record<string, number>>(initialValues);
  const [expanded, setExpanded] = useState<Record<string, boolean>>({
    "Configuration Parameters": true,
    "Geometric Parameters": true,
    "UHPC Material Properties": true,
    "Confinement Stress Parameters": true,
    "Shear Strength": true
  });

  const handleChange = (id: string, value: number) => {
    setValues(prev => ({
      ...prev,
      [id]: value
    }));
  };

  const toggleGroup = (title: string) => {
    setExpanded(prev => ({
      ...prev,
      [title]: !prev[title]
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // 将记录转换为数组
    const featuresArray = Object.values(values);
    onSubmit(featuresArray);
  };

  // 颜色主题映射
  const colorMap = {
    blue: { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-800', button: 'bg-blue-100 hover:bg-blue-200' },
    green: { bg: 'bg-green-50', border: 'border-green-200', text: 'text-green-800', button: 'bg-green-100 hover:bg-green-200' },
    purple: { bg: 'bg-purple-50', border: 'border-purple-200', text: 'text-purple-800', button: 'bg-purple-100 hover:bg-purple-200' },
    orange: { bg: 'bg-orange-50', border: 'border-orange-200', text: 'text-orange-800', button: 'bg-orange-100 hover:bg-orange-200' },
    red: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-800', button: 'bg-red-100 hover:bg-red-200' }
  };

  return (
    <div className="w-full p-4 bg-white rounded-lg shadow-md">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-800">UHPC接缝特征参数输入</h2>
        <div className="flex items-center space-x-2">
          <button
            type="button"
            onClick={() => setValues(initialValues)}
            className="text-sm px-3 py-1 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-md transition-colors"
          >
            重置默认值
          </button>
          <div className="text-sm text-gray-600">
            共 {fieldGroups.reduce((sum, group) => sum + group.fields.length, 0)} 个参数
          </div>
        </div>
      </div>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* 使用网格布局展示参数组 */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4">
          {fieldGroups.map(group => {
            const colors = colorMap[group.color as keyof typeof colorMap];
            return (
              <div key={group.title} className={`border rounded-lg overflow-hidden ${colors.border} ${colors.bg}`}>
                <button
                  type="button"
                  className={`w-full px-4 py-2 text-left font-semibold ${colors.text} ${colors.button} flex justify-between items-center`}
                  onClick={() => toggleGroup(group.title)}
                >
                  <span className="text-sm font-medium">{group.title}</span>
                  <div className="flex items-center space-x-2">
                    <span className="text-xs bg-white px-2 py-1 rounded-full">{group.fields.length}</span>
                    <span className="text-sm">{expanded[group.title] ? '▼' : '►'}</span>
                  </div>
                </button>
                
                {expanded[group.title] && (
                  <div className="p-3 bg-white border-t grid grid-cols-1 lg:grid-cols-2 gap-3">
                    {group.fields.map(field => (
                      <div key={field.id} className="space-y-1">
                        <label className="block text-gray-800 font-medium text-sm">
                          {field.label}
                        </label>
                        <div className="text-xs text-gray-600 mb-1 leading-tight">{field.description}</div>
                        
                        {field.type === 'select' ? (
                          <select
                            value={values[field.id]}
                            onChange={(e) => handleChange(field.id, Number(e.target.value))}
                            className="w-full p-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
                          >
                            {field.options && field.options.map((option: {value: number, label: string}) => (
                              <option key={option.value} value={option.value}>
                                {option.label}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="number"
                            step={field.step || 1}
                            min={field.min !== undefined ? field.min : undefined}
                            value={values[field.id]}
                            onChange={(e) => handleChange(field.id, Number(e.target.value))}
                            className="w-full p-2 text-sm border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
                          />
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            );
          })}
        </div>
        
        {/* 提交按钮 */}
        <div className="pt-4 border-t">
          <button
            type="submit"
            disabled={isLoading}
            className={`w-full py-3 px-4 rounded-lg font-semibold transition-all duration-200 ${
              isLoading
                ? 'bg-gray-400 cursor-not-allowed'
                : 'bg-blue-600 hover:bg-blue-700 text-white shadow-md hover:shadow-lg'
            }`}
          >
            {isLoading ? (
              <div className="flex items-center justify-center">
                <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full mr-2"></div>
                正在预测...
              </div>
            ) : (
              '开始预测'
            )}
          </button>
        </div>
      </form>
    </div>
  );
};

export default FeatureInput; 