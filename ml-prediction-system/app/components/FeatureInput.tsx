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
};

const FeatureInput: React.FC<FeatureInputProps> = ({ onSubmit, isLoading }) => {
  // 字段配置
  const fieldGroups: FieldGroup[] = [
    {
      title: "Configuration Parameters",
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

  return (
    <div className="w-full p-4 bg-white rounded-lg shadow-md mt-4">
      <h2 className="text-xl font-bold mb-4 text-gray-800">Input UHPC Joint Feature Data</h2>
      
      <form onSubmit={handleSubmit} className="space-y-6">
        {fieldGroups.map(group => (
          <div key={group.title} className="border rounded-lg overflow-hidden">
            <button
              type="button"
              className="w-full px-4 py-3 bg-gray-100 text-left font-semibold text-gray-800 hover:bg-gray-200 flex justify-between items-center"
              onClick={() => toggleGroup(group.title)}
            >
              <span>{group.title}</span>
              <span>{expanded[group.title] ? '▼' : '►'}</span>
            </button>
            
            {expanded[group.title] && (
              <div className="p-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                {group.fields.map(field => (
                  <div key={field.id} className="space-y-1">
                    <label className="block text-gray-800 font-medium">
                      {field.label}
                    </label>
                    <div className="text-xs text-gray-600 mb-1">{field.description}</div>
                    
                    {field.type === 'select' ? (
                      <select
                        value={values[field.id]}
                        onChange={(e) => handleChange(field.id, Number(e.target.value))}
                        className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
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
                        className="w-full p-2 border rounded focus:outline-none focus:ring-2 focus:ring-blue-500 text-gray-800"
                      />
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
        
        <button
          type="submit"
          disabled={isLoading}
          className={`w-full py-3 px-4 rounded font-semibold ${
            isLoading
              ? 'bg-gray-400 cursor-not-allowed'
              : 'bg-blue-600 hover:bg-blue-700 text-white'
          }`}
        >
          {isLoading ? 'Predicting...' : 'Start Prediction'}
        </button>
      </form>
    </div>
  );
};

export default FeatureInput; 