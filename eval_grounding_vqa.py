import json
import os
import re
import math
import io
import base64
import random
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
from openai import OpenAI
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pyarrow as pa
import pyarrow.parquet as pq
import concurrent.futures
from collections import Counter
import sys
import torch
import logging

from vision_utils import process_vision_info, smart_resize
from transformers import AutoProcessor


# 设置全局随机种子，确保完全确定性
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    from collections import Counter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    Image = None
    MATPLOTLIB_AVAILABLE = False

try:
    from transformers import Qwen2_5_VLForConditionalGeneration
    HF_AVAILABLE = True
except ImportError:
    Qwen2_5_VLForConditionalGeneration = None
    HF_AVAILABLE = False

# ===================== 通用工具函数 =====================
def get_image_before_after_size(image_path, min_pixels=0, max_pixels=1003520):
    with Image.open(image_path) as img:
        orig_width, orig_height = img.size
        new_height, new_width = smart_resize(orig_height, orig_width, min_pixels=min_pixels, max_pixels=max_pixels)
    return orig_width, orig_height, new_width, new_height

def preprocess_image(image_path, min_pixels=0, max_pixels=1003520, return_pil: bool = False):
    """
    Resizes an image, and returns either bytes or a PIL image object.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            orig_width, orig_height = img.size
            new_height, new_width = smart_resize(orig_height, orig_width, min_pixels=min_pixels, max_pixels=max_pixels)
            resized_img = img.resize((new_width, new_height))

            if return_pil:
                return resized_img, orig_width, orig_height, new_width, new_height
            
            buf = io.BytesIO()
            format_str = image_path.suffix[1:].upper()
            if format_str == 'JPG':
                format_str = 'JPEG'
            resized_img.save(buf, format=format_str)
            image_bytes = buf.getvalue()

            return image_bytes, orig_width, orig_height, new_width, new_height
    except ValueError as e:
        print(e)
        return None, None, None, None, None
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None, None, None, None


def restore_bbox_to_original(pred_bbox, orig_width, orig_height, new_width, new_height):
    """
    Restores a bounding box from preprocessed image coordinates back to original image coordinates.

    Args:
        pred_bbox (list or tuple): The predicted bounding box [x1, y1, x2, y2] on the resized image.
        orig_width (int): The original image width.
        orig_height (int): The original image height.
        new_width (int): The resized image width.
        new_height (int): The resized image height.

    Returns:
        list: The restored bounding box [x1, y1, x2, y2] in the original image's coordinate system.
    """
    if new_width == 0 or new_height == 0 or orig_width == 0 or orig_height == 0:
        # Avoid division by zero if any dimension is zero.
        return [0, 0, 0, 0]

    scale_w = new_width / orig_width
    scale_h = new_height / orig_height

    # Unpack predicted bbox
    x1_pred, y1_pred, x2_pred, y2_pred = pred_bbox

    # Apply inverse scaling
    orig_x1 = x1_pred / scale_w if scale_w != 0 else 0
    orig_y1 = y1_pred / scale_h if scale_h != 0 else 0
    orig_x2 = x2_pred / scale_w if scale_w != 0 else 0
    orig_y2 = y2_pred / scale_h if scale_h != 0 else 0

    orig_bbox_float = np.array([orig_x1, orig_y1, orig_x2, orig_y2])
    
    # Round to nearest integer
    orig_bbox_int = np.round(orig_bbox_float).astype(int)
    
    # Clip coordinates to ensure they are within the original image boundaries
    orig_bbox_int[[0, 2]] = np.clip(orig_bbox_int[[0, 2]], 0, orig_width - 1)
    orig_bbox_int[[1, 3]] = np.clip(orig_bbox_int[[1, 3]], 0, orig_height - 1)
    
    return orig_bbox_int.tolist()


def normalize_text(text: str) -> str:
    """
    Normalizes text by lowercasing, removing punctuation and extra whitespace.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text

def semantic_match(question: str, pred_result: str, gt_result: str, verbose: bool = False) -> bool:

    """
    语义匹配函数，判断两个结果是否语义相同
    
    Args:
        question: 问题
        pred_result: 预测结果
        gt_result: 真实结果
        
    Returns:
        是否语义匹配
    """
    # 首先进行基础的文本匹配检查（快速过滤）
    normalized_pred = normalize_text(pred_result)
    normalized_gt = normalize_text(gt_result)
    
    # 处理空字符串情况
    if not normalized_pred and not normalized_gt:
        return True
    if not normalized_pred or not normalized_gt:
        return False
    
    # 完全匹配
    if normalized_pred == normalized_gt:
        return True
    
    # can use the llm-as-judge to judge the semantic match
    return False


def process_grounding_label(bbox_4pt, orig_width, orig_height, new_width, new_height):
    """
    Scales the bounding box coordinates to match the resized image. 
    """
    if new_width is None or new_height is None or orig_width == 0 or orig_height == 0:
        return [0, 0, 0, 0]

    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox_4pt
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    # Clip coordinates to ensure they are within the new image boundaries
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]


def calculate_iou(box_gt, box_pred):
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.

    Args:
        box_gt (list): The ground truth bounding box [x1, y1, x2, y2].
        box_pred (list): The predicted bounding box [x1, y1, x2, y2].

    Returns:
        float: The IoU value.
    """
    # Determine the coordinates of the intersection rectangle
    x_left = max(box_gt[0], box_pred[0])
    y_top = max(box_gt[1], box_pred[1])
    x_right = min(box_gt[2], box_pred[2])
    y_bottom = min(box_gt[3], box_pred[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Compute the area of intersection
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute the area of both bounding boxes
    gt_area = (box_gt[2] - box_gt[0]) * (box_gt[3] - box_gt[1])
    pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])

    # Compute the union area
    union_area = gt_area + pred_area - intersection_area

    # Compute the IoU
    iou = intersection_area / union_area if union_area > 0 else 0.0
    return iou

def create_vqa_prompt(question):
    """
    构造VQA任务的prompt结构，返回结构化字典。
    """
    # system_prompt = '你是一个视觉问答专家，请根据用户问题和图片内容，给出简洁准确的答案。'
    # user_prompt = f'<image>\n用户问题：{question}'
    system_prompt = 'You are a helpful assistant.'
    user_prompt = f'<image>\n请回答: {question}'
    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'question': question,
        'task_type': 'vqa',
        'prompt_list': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    }

def create_vqa_prompt_v2(item: Dict[str, Any]):
    """
    根据给定的item，使用包含背景信息的新模板构造VQA prompt。
    """
    vg_question_list = item.get('vg_question_list', [])
    vqa_question_list = item.get('vqa_question_list', [])
    question = item.get('question', '')

    vg_list_str = '、'.join(vg_question_list) if vg_question_list else "无"
    vqa_list_str = '、'.join(vqa_question_list) if vqa_question_list else "无"

    prompt_template = '''\
现在有：
1、一张图片。
2、一个图中出现的部分物体清单（少量情况下，可能没有）。
3、一个针对该图片的问题清单（如果清单中有多个问题，这些问题之间可能是有关联的，一个问题的答案可能在另一个问题中有线索）。

需要你根据图片、图中出现的部分物体清单、问题清单，来回答问题。
图中出现的部分物体清单是：{vg_question_list}
针对该图片的问题清单是：{vqa_question_list}

现在请根据这些信息回答这个问题：{question}
请用中文回答, 回答内容不要包含任何其他内容, 直接返回答案'''

    user_prompt = prompt_template.format(
        vg_question_list=vg_list_str,
        vqa_question_list=vqa_list_str,
        question=question
    )

    # system_prompt = '你是一个聪明的助手，请根据提供的信息回答问题。'
    system_prompt = 'You are a helpful assistant.'
    user_prompt_with_image = f'<image>\n{user_prompt}'

    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt_with_image,
        'question': question,
        'task_type': 'vqa',
        'prompt_list': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt_with_image}
        ]
    }

def create_vqa_prompt_v2_hf(item: Dict[str, Any]):
    """
    根据给定的item，使用包含背景信息的新模板构造VQA prompt for huggingface。
    """
    prompt_data = create_vqa_prompt_v2(item)
    user_prompt_without_image_tag = prompt_data['user_prompt'].replace('<image>\n', '')
    
    return {
        'system_prompt': prompt_data['system_prompt'],
        'user_prompt': user_prompt_without_image_tag,
        'question': prompt_data['question'],
        'task_type': prompt_data['task_type'],
    }

# 完全参考eval_mars_grounding.py实现，便于grounding任务prompt标准化

def create_grounding_prompt(question):
    """
    构造grounding任务的prompt结构，返回结构化字典。
    """
    system_prompt = 'You are a helpful assistant.'
    user_prompt = f'<image>\nLocate {question} in this image and output the bbox coordinates in JSON format.'
    return {
        'system_prompt': system_prompt,
        'user_prompt': user_prompt,
        'question': question,
        'task_type': 'visual_grounding',
        'prompt_list': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    }

def create_grounding_prompt_hf(question):
    """
    构造grounding任务的prompt结构，返回结构化字典 for huggingface。
    """
    prompt_data = create_grounding_prompt(question)
    user_prompt_without_image_tag = prompt_data['user_prompt'].replace('<image>\n', '')

    return {
        'system_prompt': prompt_data['system_prompt'],
        'user_prompt': user_prompt_without_image_tag,
        'question': prompt_data['question'],
        'task_type': prompt_data['task_type'],
    }

# ===================== 评测基类 =====================

class MarsBaseEvaluator(ABC):
    def __init__(self, json_path: str, task_type: str = 'grounding', debug=False):
        self.json_path = json_path
        self.task_type = task_type
        self.debug = debug
        self.data = self.load_data()

    def load_data(self) -> List[Dict[str, Any]]:
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if self.debug:
            data = data[:10]
        return data

    @abstractmethod
    def preprocess_item(self, item, index, image_base_dir=None, min_pixels=0, max_pixels=1003520):
        """
        子类实现：对单条样本做结构化预处理（如图片bytes、prompt、bbox等）。
        """
        pass

    def infer_batch(self, model, batch: List[Dict[str, Any]], model_type: str, tokenizer, client, model_name, image_base_dir, args: Any) -> List[Any]:
        if model_type == 'hf':
            return self._infer_batch_hf(model, batch, tokenizer, image_base_dir, args)
        elif model_type == 'client':
            return self._infer_batch_client(client, batch, model_name, image_base_dir, args)
        else:
            raise ValueError(f"不支持的model_type: {model_type}")

    @staticmethod
    def _make_request_with_retry(client, params, timeout=60, retries=3, question=""):
        for attempt in range(retries):
            try:
                return client.chat.completions.create(**params, timeout=timeout)
            except Exception as e:
                print(f"Request for '{question}' failed on attempt {attempt + 1}/{retries}: {e}")
                if attempt + 1 == retries:
                    print(f"All retries failed for '{question}'.")
                    raise
        return None

    def _infer_batch_hf(self, model, batch, tokenizer, image_base_dir, args):
        all_preds = []
        batch_size = 1 # bug for > 1 due to the qwen2.5-vl bug,  https://github.com/huggingface/transformers/issues/37606
        for i in tqdm(range(0, len(batch), batch_size), desc=f"HF {self.task_type.capitalize()} Inference"):
            chunk = batch[i:i + batch_size]
            
            all_messages = []
            metadata_list = []
            for item in chunk:
                messages, metadata = self._prepare_hf_messages(item, image_base_dir, args)
                all_messages.append(messages)
                metadata_list.append(metadata)

            texts, all_image_inputs = [], []
            for messages in all_messages:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, _ = process_vision_info(messages)
                texts.append(text)
                all_image_inputs.append(image_inputs)
            
            inputs = tokenizer(text=texts, images=all_image_inputs, padding=True, return_tensors="pt").to(model.device)
            
            generate_kwargs = {"max_new_tokens": 1024, "do_sample": False}
            generated_ids = model.generate(**inputs, **generate_kwargs)
            effective_input_ids = inputs.input_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(effective_input_ids, generated_ids)
            ]
            output_texts = tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

            chunk_results = self._parse_hf_output(output_texts, metadata_list)
            all_preds.extend(chunk_results)
        return all_preds

    def _infer_batch_client(self, client, batch, model_name, image_base_dir, args):
        results = [None] * len(batch)
        
        max_workers = args.num_workers if args.num_workers is not None else (32 if self.task_type == 'grounding' else 5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._infer_single_client_item, (i, item), client, model_name, image_base_dir, args, results) for i, item in enumerate(batch)]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(batch), desc=f"{self.task_type.capitalize()}推理(client)"):
                future.result()

        return results

    @abstractmethod
    def _prepare_hf_messages(self, item: Dict[str, Any], image_base_dir, args) -> Tuple[List[Dict], Any]:
        """Prepares HF messages and metadata for a single item."""
        pass

    @abstractmethod
    def _parse_hf_output(self, output_texts: List[str], metadata: List[Any]) -> List[Any]:
        """Parses the output from an HF batch."""
        pass
    
    @abstractmethod
    def _infer_single_client_item(self, item_with_index, client, model_name, image_base_dir, args, results_list) -> None:
        """Performs inference for a single item using the client API and stores result in results_list."""
        pass

    @abstractmethod
    def parse_result(self, item: Dict[str, Any]) -> Any:
        pass

    @abstractmethod
    def evaluate(self, preds: List[Any], gts: List[Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    def process_results(self, preds: List[Any], gts: Optional[List[Any]], original_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        处理预测和真值，以生成详细结果和评估指标。
        返回结果项列表和指标字典。
        """
        pass


    def handle_inference_task(self, args):
        """
        Handles the entire inference, evaluation, and result processing task.
        """
        model_config = setup_inference_engine(args)
        
        data = self.data
        print(f"Total samples: {len(data)}")

        # Automatically detect if ground truth is available
        has_ground_truth = False
        if data and 'result' in data[0]:
            has_ground_truth = True
        print(f"has_ground_truth: {has_ground_truth}")

        all_results, metrics = run_inference_and_save(self, data, model_config, has_ground_truth=has_ground_truth, args=args)

# ===================== VQA评测器 =====================

class MarsVQAEvaluator(MarsBaseEvaluator):
    def __init__(self, json_path: str, prompt_version: str = 'v1', debug=False):
        super().__init__(json_path, task_type='vqa', debug=debug)
        self.prompt_version = prompt_version

    def preprocess_item(self, item, index, image_base_dir=None, min_pixels=0, max_pixels=1003520):
        from pathlib import Path
        image_path_str = item['image_path'].replace('\\\\', '/').replace('\\', '/')
        image_path = Path(image_base_dir) / Path(image_path_str)
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}, skipping item.")
            return None
        image_bytes, orig_width, orig_height, new_width, new_height = preprocess_image(image_path, min_pixels=min_pixels, max_pixels=max_pixels)
        if image_bytes is None:
            return None
        if self.prompt_version == 'v2':
            prompt_data = create_vqa_prompt_v2(item)
        else:
            prompt_data = create_vqa_prompt(item['question'])
        record = {
            'images': [{'bytes': image_bytes, 'path': item['image_path']}],
            'prompt': prompt_data['prompt_list'],
            'question': item['question'],
            'orig_width': orig_width,
            'orig_height': orig_height,
            'new_width': new_width,
            'new_height': new_height,
            'index': index,
            'raw': item
        }
        return record


    def _prepare_hf_messages(self, item: Dict[str, Any], image_base_dir, args) -> Tuple[List[Dict], Any]:
        image_path_str = item['image_path'].replace('\\', '/')
        if image_base_dir and not os.path.isabs(image_path_str):
            full_image_path = os.path.join(image_base_dir, image_path_str)
        else:
            full_image_path = image_path_str

        if self.prompt_version == 'v2':
            prompt_data = create_vqa_prompt_v2_hf(item)
            user_prompt_text = prompt_data['user_prompt']
        else: # v1
            prompt_data = create_vqa_prompt(item['question'])
            user_prompt_text = prompt_data['user_prompt'].replace('<image>\n', '')

        messages = [
            {"role": "system", "content": prompt_data['system_prompt']},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image_path, 
                    'max_pixels': args.max_pixels, 
                    'min_pixels': args.min_pixels},
                    {"type": "text", "text": user_prompt_text},
                ],
            }
        ]
        return messages, None

    def _parse_hf_output(self, output_texts: List[str], metadata: List[Any]) -> List[Any]:
        return output_texts

    def _infer_single_client_item(self, item_with_index, client, model_name, image_base_dir, args, results_list) -> None:
        idx, item = item_with_index
        try:
            image_path_str = item['image_path'].replace('\\', '/')
            question = item['question']
            
            if self.prompt_version == 'v2':
                prompt_data = create_vqa_prompt_v2(item)
            else:
                prompt_data = create_vqa_prompt(question)

            image_path = Path(image_base_dir) / image_path_str
            image_bytes, _, _, _, _ = preprocess_image(image_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            if image_bytes is None:
                results_list[idx] = None
                return

            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            text_prompt = prompt_data['user_prompt'].replace('<image>', '')
            messages = [
                {"role": "system", "content": prompt_data['system_prompt']},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": text_prompt},
                ]},
            ]
            params = {
                "model": model_name, "messages": messages, "temperature": 0, "max_tokens": 1024,
                "stop": ["<|im_end|>\n".strip()],
            }

            response = self._make_request_with_retry(client, params, question=item.get('question'))
            results_list[idx] = response.choices[0].message.content
        except Exception as e:
            print(f"Error inferring item {item.get('question')}: {e}")
            results_list[idx] = None

    def parse_result(self, item: Dict[str, Any]) -> str:
        return item['result']

    def evaluate(self, preds: List[str], gts: List[str], questions: List[str] = None) -> Dict[str, Any]:
        if questions is None:
            raise ValueError("Questions must be provided for semantic matching.")
        
        if len(preds) != len(questions):
            raise ValueError("Length of predictions and questions must be the same for semantic matching.")

        correct = 0
        for i, (p, g) in enumerate(zip(preds, gts)):
            if semantic_match(questions[i], p, g, verbose=True):
                correct += 1
        
        total = len(gts)
        acc = correct / total if total > 0 else 0
        return {'accuracy': acc, 'total': total, 'correct': correct}

    def process_results(self, preds: List[Any], gts: Optional[List[Any]], original_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        all_results = []
        metrics = {}

        if gts:
            questions = [item['question'] for item in original_data]
            metrics = self.evaluate(preds, gts, questions=questions)

        for i, item in enumerate(original_data):
            result_item = {
                'image_path': item['image_path'],
                'question': item['question'],
            }
            result_item['prediction'] = preds[i]
            if gts:
                result_item['ground_truth'] = gts[i]
                result_item['is_correct'] = semantic_match(item['question'], preds[i], gts[i])
            all_results.append(result_item)
        
        return all_results, metrics

# ===================== Grounding评测器 =====================

class MarsGroundingEvaluator(MarsBaseEvaluator):
    def __init__(self, json_path: str, debug=False):
        super().__init__(json_path, task_type='grounding', debug=debug)

    def preprocess_item(self, item, index, image_base_dir=None, min_pixels=0, max_pixels=1003520):
        from pathlib import Path
        image_path_str = item['image_path'].replace('\\\\', '/').replace('\\', '/')
        image_path = Path(image_base_dir) / Path(image_path_str)
        if not image_path.exists():
            print(f"Warning: Image not found at {image_path}, skipping item.")
            return None
        bbox_2pt = item.get('result')
        bbox_4pt = [bbox_2pt[0][0], bbox_2pt[0][1], bbox_2pt[1][0], bbox_2pt[1][1]] if bbox_2pt else None
        image_bytes, orig_width, orig_height, new_width, new_height = preprocess_image(image_path, min_pixels=min_pixels, max_pixels=max_pixels)
        if image_bytes is None:
            return None
        new_bbox_4pt = process_grounding_label(bbox_4pt, orig_width, orig_height, new_width, new_height) if bbox_4pt else None
        prompt_data = create_grounding_prompt(item['question'])
        record = {
            'images': [{'bytes': image_bytes, 'path': item['image_path']}],
            'prompt': prompt_data['prompt_list'],
            'question': item['question'],
            'resized_bbox': new_bbox_4pt,
            'orig_width': orig_width,
            'orig_height': orig_height,
            'new_width': new_width,
            'new_height': new_height,
            'index': index,
            'raw': item
        }
        return record


    def _prepare_hf_messages(self, item: Dict[str, Any], image_base_dir, args) -> Tuple[List[Dict], Any]:
        image_path_str = item['image_path'].replace('\\', '/')
        if image_base_dir and not os.path.isabs(image_path_str):
            full_image_path = os.path.join(image_base_dir, image_path_str)
        else:
            full_image_path = image_path_str
        
        orig_width, orig_height, new_width, new_height = get_image_before_after_size(full_image_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
        metadata = (orig_width, orig_height, new_width, new_height)

        prompt_data = create_grounding_prompt_hf(item['question'])
        messages = [
            {"role": "system", "content": prompt_data['system_prompt']},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image_path, 
                    'max_pixels': args.max_pixels, 
                    'min_pixels': args.min_pixels},
                    {"type": "text", "text": prompt_data['user_prompt']},
                ],
            }
        ]
        return messages, metadata

    def _parse_hf_output(self, output_texts: List[str], metadata: List[Any]) -> List[Any]:
        chunk_results = []
        for i in range(len(output_texts)):
            dims = metadata[i]
            bbox = parse_and_restore_bbox(output_texts[i], dims)
            chunk_results.append(bbox)
        return chunk_results

    def _infer_single_client_item(self, item_with_index, client, model_name, image_base_dir, args, results_list) -> None:
        idx, item = item_with_index
        try:
            image_path_str = item['image_path'].replace('\\', '/')
            question = item['question']
            prompt_data = create_grounding_prompt(question)

            image_path = Path(image_base_dir) / image_path_str
            image_bytes, orig_width, orig_height, new_width, new_height = preprocess_image(image_path, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            text_prompt = prompt_data['user_prompt'].replace('<image>', '')
            messages = [
                {"role": "system", "content": prompt_data['system_prompt']},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    {"type": "text", "text": text_prompt},
                ]},
            ]
            params = {
                "model": model_name, "messages": messages, 
                "temperature": 0,
                "max_tokens": 10240,
                # "stop": ["<|im_end|>\n".strip()],
                "seed": 42,
            }

            response = self._make_request_with_retry(client, params, question=item.get('question'), timeout=30, retries=10)
            bbox = parse_and_restore_bbox(response.choices[0].message.content, (orig_width, orig_height, new_width, new_height))
            results_list[idx] = bbox
        except Exception as e:
            print(f"Error inferring item {item.get('question')}: {e}")
            results_list[idx] = None

    def parse_result(self, item: Dict[str, Any]) -> Optional[List[int]]:
        bbox_2pt = item.get('result')
        if not bbox_2pt:
            return None
        return [bbox_2pt[0][0], bbox_2pt[0][1], bbox_2pt[1][0], bbox_2pt[1][1]]

    def evaluate(self, preds: List[Optional[List[int]]], gts: List[Optional[List[int]]]) -> Dict[str, Any]:
        iou_scores = []
        for pred, gt in zip(preds, gts):
            iou = 0.0
            if pred and gt:
                iou = calculate_iou(gt, pred)
            iou_scores.append(iou)

        total = len(gts)
        if total == 0:
            return {'IoU@0.5': 0, 'average_iou': 0, 'total': 0, 'correct': 0}

        correct = sum(1 for iou in iou_scores if iou > 0.5)
        acc = correct / total
        avg_iou = np.mean([iou for iou in iou_scores if iou is not None])
        
        return {'IoU@0.5': acc, 'average_iou': float(avg_iou), 'total': total, 'correct': correct}

    def process_results(self, preds: List[Any], gts: Optional[List[Any]], original_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        all_results = []
        metrics = {}

        if gts:
            metrics = self.evaluate(preds, gts)

        for i, item in enumerate(original_data):
            result_item = {
                'image_path': item['image_path'],
                'question': item['question'],
            }
            result_item['predicted_bbox'] = preds[i]
            if gts:
                result_item['ground_truth_bbox'] = gts[i]
                iou = 0.0
                if preds[i] and gts[i]:
                    iou = calculate_iou(gts[i], preds[i])
                result_item['iou'] = iou
            all_results.append(result_item)
        
        return all_results, metrics

# ========== 推理后处理与评测工具 ==========

def parse_and_restore_bbox(generated_text, dimensions, verbose=False, image_name=""):
    """
    Parses model output to find a bounding box, and restores it to original image coordinates.
    """
    orig_width, orig_height, new_width, new_height = dimensions
    predicted_bbox = None
    # Enhanced regex to find JSON object or array, and handle surrounding text
    match = re.search(r'```json\s*([\[\{].*?[\]\}])\s*```|([\[\{].*?[\]\}])', generated_text, re.DOTALL)
    
    json_text = None
    if match:
        # The regex has two capturing groups, one for markdown blocks, one for raw json.
        # `or` returns the first non-empty group.
        json_text = match.group(1) or match.group(2)

    if json_text:
        predicted_bbox_resized = None
        try:
            # Try to fix common formatting errors from the model
            cleaned_json_text = re.sub(r'(?<=\])\s*"\s*(?=\})', '', json_text.strip())
            # 修复括号不匹配的问题，将 ")" 替换为 "}"
            cleaned_json_text = re.sub(r'\)\s*$', '}', cleaned_json_text)
            predicted_data = json.loads(cleaned_json_text)
            
            bbox_data = None
            if isinstance(predicted_data, list):
                if len(predicted_data) > 0 and isinstance(predicted_data[0], dict):
                    bbox_data = predicted_data[0]
            elif isinstance(predicted_data, dict):
                bbox_data = predicted_data

            if bbox_data:
                # 支持两种格式：直接是bbox_2d列表，或者包含在bbox_2d字段中
                if "bbox_2d" in bbox_data:
                    predicted_bbox_resized = bbox_data.get("bbox_2d")
                elif len(bbox_data) >= 4 and all(isinstance(v, (int, float)) for v in list(bbox_data.values())[:4]):
                    # 如果bbox_data直接包含坐标值，尝试提取前4个数值
                    values = list(bbox_data.values())
                    if len(values) >= 4:
                        predicted_bbox_resized = values[:4]

            if predicted_bbox_resized and isinstance(predicted_bbox_resized, list) and len(predicted_bbox_resized) == 4:
                if verbose:
                    print(f"Model predicted bbox (on resized image): {predicted_bbox_resized}")
                
                predicted_bbox = restore_bbox_to_original(
                    predicted_bbox_resized, orig_width, orig_height, new_width, new_height
                )
                if verbose:
                    print(f"Restored bbox (on original image): {predicted_bbox}")
            else:
                msg = f"Could not find a valid 'bbox_2d' list with 4 elements in the JSON: {json_text}"
                if image_name: msg += f" for {image_name}"
                print(f"Warning: {msg}")
        except json.JSONDecodeError:
            msg = f"Could not decode JSON from response: {generated_text}, {predicted_bbox_resized}, {orig_width}, {orig_height}, {new_width}, {new_height}"
            if image_name: msg += f" for {image_name}"
            print(f"Warning: {msg}")
    else:
        msg = "Could not find a valid JSON object or array in the model's response"
        if image_name: msg += f" for {image_name}"
        print(f"Warning: {msg}")
        print(f"Warning: Generated text: {generated_text}")
    return predicted_bbox


def run_inference_and_save(evaluator, data, model_config, has_ground_truth, args):
    """
    执行推理、评估并保存结果。
    """
    llm, tokenizer, client = model_config
    
    if has_ground_truth:
        gts = [evaluator.parse_result(item) for item in data]
    else:
        gts = None

    preds = evaluator.infer_batch(
        model=llm, 
        batch=data, 
        model_type=args.inference_mode, 
        tokenizer=tokenizer, 
        client=client,
        model_name=args.model_name,
        image_base_dir=args.image_base_dir,
        args=args,
    )

    all_results, metrics = evaluator.process_results(preds, gts, data)

    if has_ground_truth:
        print("\n--- Evaluation Metrics ---")
        print(metrics)
        print("--------------------------\n")
    
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=4)
    print(f"Saved detailed inference results to: {args.output_path}")

    submission_path = args.output_path.replace('.json', '_submission.json')
    generate_submission_file(all_results, submission_path, args.task)
    
    return all_results, metrics

def generate_submission_file(all_results, output_path, task):
    """
    生成submission文件，格式与主脚本一致。
    """
    submission_data = []
    for result in all_results:
        submission_item = {
            "image_path": result['image_path'],
            "question": result['question'],
        }
        if task == 'grounding':
            predicted_bbox = result.get('predicted_bbox')
            if not predicted_bbox and 'model_responses' in result and result['model_responses']:
                predicted_bbox = result['model_responses'][0]  # Take the first one for k>1

            if predicted_bbox and predicted_bbox != "N/A" and len(predicted_bbox) == 4:
                bbox_2pt = [
                    [predicted_bbox[0], predicted_bbox[1]],
                    [predicted_bbox[2], predicted_bbox[3]]
                ]
            else:
                bbox_2pt = [[0, 0], [0, 0]]
            submission_item['result'] = bbox_2pt
        elif task == 'vqa':
            prediction = result.get('prediction')
            if prediction is None and 'model_responses' in result and result['model_responses']:
                prediction = result['model_responses'][0]  # Take the first one for k>1
            submission_item['result'] = prediction if prediction is not None else 'N/A'
        
        submission_data.append(submission_item)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(submission_data, f, ensure_ascii=False, indent=4)
    print(f"Saved submission file to: {output_path}")
    print(f"Total samples in submission: {len(submission_data)}")


# ===================== 主入口 =====================

def setup_inference_engine(args):
    """
    Initializes and returns the inference engine (HuggingFace model or API client).
    """
    llm, tokenizer, client = None, None, None
    if args.inference_mode == 'hf':
        if not HF_AVAILABLE:
            raise ImportError("huggingface transformers is not available. Please install it.")
        if args.model_path is None:
            raise ValueError('请指定--model_path')

        if args.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            print(f"CUDA_VISIBLE_DEVICES set to '{args.gpu_ids}' for HF mode.")

        print(f"加载HuggingFace模型: {args.model_path}")
        tokenizer = AutoProcessor.from_pretrained(args.model_path, padding_side='left', trust_remote_code=True)
        llm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("加载模型完毕")
    else: # client mode
        if args.port is None:
            port = 18901
        else:
            port = args.port
        client = OpenAI(api_key='EMPTY', base_url=f'http://0.0.0.0:{port}/v1')
        print(f"使用API client模式, base_url: http://0.0.0.0:{port}/v1")
    
    print("Inference engine setup complete.")
    return llm, tokenizer, client



def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='../data/mars/VQA-SA/VQA-SA-val_v1_1_with_context.json')
    parser.add_argument('--task', type=str, choices=['vqa', 'grounding'], default='vqa')
    parser.add_argument('--prompt_version', type=str, choices=['v1', 'v2'], default='v2', help='VQA prompt的版本, v1为旧版, v2为新版')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--model_path', type=str, default='Zach996/ActiveAlphaAgent-VG-RS', help='llm模型路径')
    parser.add_argument('--image_base_dir', type=str, default='../data/mars/VG-RS', help='图片根目录')
    parser.add_argument('--inference_mode', type=str, default='client', choices=['client', 'hf'], help='推理模式')
    parser.add_argument('--model_name', type=str, default=None, help='模型名称')
    parser.add_argument('--output_path', type=str, default=None, help='输出json文件路径')
    parser.add_argument('--num_workers', type=int, default=None, help='Number of worker threads for data processing.')
    parser.add_argument('--port', type=int, default=None, help='API client port')
    parser.add_argument('--gpu_ids', type=str, default=None, help='Comma separated list of GPU IDs to use for HF mode (e.g., "0,1,2,3")')
    parser.add_argument('--min_pixels', type=int, default=0, help='Minimum number of pixels for smart resize.')
    parser.add_argument('--max_pixels', type=int, default=1003520, help='Maximum number of pixels for smart resize.')
    args = parser.parse_args()
    print(args)
    DEBUG = False

    if args.output_path is None:
        json_file_name = os.path.splitext(os.path.basename(args.json_path))[0]
    
        if args.model_name is not None:
            json_file_name = f"{json_file_name}_{args.model_name}"

        args.output_path = os.path.join(args.output_dir, f"{json_file_name}_sft.jsonl")
        print(f"输出路径未指定，将使用默认路径: {args.output_path}")

    # --- Task Dispatching ---
    if args.task in ['vqa', 'grounding']:
        print(f'task is {args.task}')
        if args.task == 'vqa':
            evaluator = MarsVQAEvaluator(args.json_path, prompt_version=args.prompt_version, debug=DEBUG)
        else: # grounding
            evaluator = MarsGroundingEvaluator(args.json_path, debug=DEBUG)
        evaluator.handle_inference_task(args)

    else:
        raise ValueError(f"Unknown task: {args.task}")


if __name__ == '__main__':
    main()
