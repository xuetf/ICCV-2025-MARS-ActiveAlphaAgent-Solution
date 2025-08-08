#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
处理任意格式与VQA-SA-question.json相同的JSON文件，为其添加vqa_question_list和vg_question_list字段
"""

import json
import argparse
import sys
from collections import defaultdict
import os

def load_question_data(file_path, description):
    """加载问题数据文件"""
    print(f"正在读取{description}文件: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"错误: 文件 {file_path} 不存在")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"{description}数据: {len(data)} 条")
        return data
    except Exception as e:
        print(f"错误: 读取文件 {file_path} 失败: {e}")
        return None

def process_questions_with_context(input_file, output_file, vqa_file="data/VQA-SA-question.json", vg_file="data/VG-RS-question.json", save_result=False):
    """处理输入文件，为其添加VQA和VG问题列表
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        vqa_file: VQA-SA问题文件路径
        vg_file: VG-RS问题文件路径
        save_result: 是否保存input_file中的result字段到output_file
    """
    
    # 加载输入文件
    input_data = load_question_data(input_file, "输入")
    if input_data is None:
        return False
    
    # 加载VQA-SA数据
    vqa_data = load_question_data(vqa_file, "VQA-SA")
    if vqa_data is None:
        return False
    
    # 加载VG-RS数据
    vg_data = load_question_data(vg_file, "VG-RS")
    if vg_data is None:
        return False
    
    # 验证输入文件格式
    if not input_data or not isinstance(input_data, list):
        print("错误: 输入文件格式不正确，应该是JSON数组")
        return False
    
    if not all(isinstance(item, dict) and 'image_path' in item and 'question' in item for item in input_data):
        print("错误: 输入文件格式不正确，每个项目应该包含image_path和question字段")
        return False
    
    # 按图片路径分组VQA-SA问题
    vqa_image_questions = defaultdict(list)
    for item in vqa_data:
        image_path = item['image_path']
        question = item['question']
        vqa_image_questions[image_path].append(question)
    
    # 按图片路径分组VG-RS问题
    vg_image_questions = defaultdict(list)
    for item in vg_data:
        image_path = item['image_path']
        question = item['question']
        vg_image_questions[image_path].append(question)
    
    print(f"VQA-SA图片数量: {len(vqa_image_questions)}")
    print(f"VG-RS图片数量: {len(vg_image_questions)}")
    
    # 获取输入文件的图片路径
    input_images = set(item['image_path'] for item in input_data)
    print(f"输入文件图片数量: {len(input_images)}")
    
    # 统计匹配情况
    vqa_matches = len(input_images & set(vqa_image_questions.keys()))
    vg_matches = len(input_images & set(vg_image_questions.keys()))
    print(f"与VQA-SA匹配的图片数量: {vqa_matches}")
    print(f"与VG-RS匹配的图片数量: {vg_matches}")
    
    # 创建新的数据结构
    result = []
    
    for item in input_data:
        image_path = item['image_path']
        question = item['question']
        
        # 获取该图片的VQA-SA问题列表
        vqa_question_list = vqa_image_questions.get(image_path, [])
        
        # 获取该图片的VG-RS问题列表
        vg_question_list = vg_image_questions.get(image_path, [])
        
        # 创建新的数据项
        new_item = {
            "image_path": image_path,
            "question": question,
            "vqa_question_list": vqa_question_list,
            "vg_question_list": vg_question_list
        }
        
        # 如果save_result为True且原数据中有result字段，则保存result字段
        if save_result and 'result' in item:
            new_item['result'] = item['result']
        
        result.append(new_item)
    
    # 保存结果
    print(f"正在保存结果到: {output_file}")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"错误: 保存文件 {output_file} 失败: {e}")
        return False
    
    print(f"处理完成！共生成 {len(result)} 条数据")
    
    # 验证结果
    print("\n验证结果:")
    print(f"原始数据条数: {len(input_data)}")
    print(f"处理后数据条数: {len(result)}")
    
    # 统计有VQA和VG问题的数据数量
    vqa_with_context = sum(1 for item in result if item['vqa_question_list'])
    vg_with_context = sum(1 for item in result if item['vg_question_list'])
    print(f"有VQA问题的数据数量: {vqa_with_context}")
    print(f"有VG问题的数据数量: {vg_with_context}")
    
    # 检查几个示例
    print("\n示例数据:")
    for i, item in enumerate(result[:3]):
        print(f"\n示例 {i+1}:")
        print(f"  图片路径: {item['image_path']}")
        print(f"  当前问题: {item['question']}")
        print(f"  VQA问题数量: {len(item['vqa_question_list'])}")
        print(f"  VG问题数量: {len(item['vg_question_list'])}")
        if item['vqa_question_list']:
            print(f"  VQA问题示例: {item['vqa_question_list'][:2]}")
        if item['vg_question_list']:
            print(f"  VG问题示例: {item['vg_question_list'][:2]}")
    
    return True

def main():
    # 确定输入和输出文件路径
    input_file = "data/VQA-SA/VQA-SA-question.json"
    output_file = "data/VQA-SA/VQA-SA-question_with_context.json"
    # 确定VQA和VG文件路径
    vqa_file = "data/VQA-SA/VQA-SA-question.json"
    vg_file = "data/VG-RS/VG-RS-question.json"
    
    # 处理文件
    success = process_questions_with_context(input_file, output_file, vqa_file, vg_file, save_result=True)
    


if __name__ == "__main__":
    main() 