import base64
import os
import json
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm
import numpy as np
from PIL import Image
from io import BytesIO
from openai import OpenAI
from pathlib import Path
from vision_utils import process_vision_info



def _encode_frame(frame_array: np.ndarray) -> str:
    """将单帧 numpy 数组编码为 base64 字符串。"""
    img = Image.fromarray(frame_array)
    output_buffer = BytesIO()
    img.save(output_buffer, format="jpeg")
    byte_data = output_buffer.getvalue()
    return base64.b64encode(byte_data).decode("utf-8")


def prepare_message_for_vllm(messages):
    """
    使用 qwen_vl_utils.process_vision_info 提取视频帧, 并为 vLLM API 调用准备消息。
    vLLM 可以通过 `extra_body` 接收 fps 等参数以进行更精确的处理。
    此函数将视频输入（本地路径或URL）转换为 vLLM 期望的内联 base64 帧格式。
    """
    vllm_messages, video_kwargs_list = [], []
    
    for message in messages:
        # 只处理 user 角色且 content 是列表的消息
        if message["role"] != "user" or not isinstance(message["content"], list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        has_video = False
        for part in message["content"]:
            if part.get("type") == "video":
                has_video = True
                # `process_vision_info` 需要一个嵌套列表的输入
                temp_message_for_processing = [{'role': 'user', 'content': [part]}]
                
                # 调用工具函数处理视频信息
                start_time = time.time()
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    temp_message_for_processing, 
                    return_video_kwargs=True
                )
                duration = time.time() - start_time
                print(f'process_vision_info 处理视频信息完成, 耗时: {duration:.2f}s, video_kwargs_list: {video_kwargs}, video_inputs_len: {len(video_inputs)}')

                if video_inputs is None or not video_inputs:
                    raise ValueError("使用 process_vision_info 未能从视频中提取任何帧。")

                # 将 Tensor 转换为 Numpy 数组以便用 PIL 处理
                # (T, C, H, W) -> (T, H, W, C)
                video_tensor = (video_inputs.pop()).permute(0, 2, 3, 1).numpy().astype(np.uint8)
                video_kwargs_list.append(video_kwargs)

                # 使用线程池并行将每一帧编码为 base64
                # 对于CPU密集型任务，将worker数设置为CPU核心数通常是最佳实践
                print(f'开始将视频帧转换为 base64 格式')
                start_time = time.time()
                with ThreadPoolExecutor(max_workers=16) as executor:
                    base64_frames = list(executor.map(_encode_frame, video_tensor))
                duration = time.time() - start_time
                print(f'将视频帧转换为 base64 格式完成, 耗时: {duration:.2f}s')
                # 构建 vLLM 特定的 video_url 格式
                # 注意：这里我们用 `video_url` 和 `data:video/jpeg` 来告诉 vLLM
                # 这是一个已经处理好的帧序列，vLLM 不应再尝试自行解码视频。
                new_part = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
                new_content_list.append(new_part)
            else:
                new_content_list.append(part)
        
        # 用新的 content 更新 message
        if has_video:
            message["content"] = new_content_list
        
        vllm_messages.append(message)

    # 合并所有视频的 kwargs，这里假设只有一个视频
    final_video_kwargs = {}
    for kwargs in video_kwargs_list:
        final_video_kwargs.update(kwargs)

    return vllm_messages, final_video_kwargs


def run_inference_with_utils(client, model_name: str, video_path: str, question: str, fps: float = 2.0, timeout: int = 60*10, retries: int = 10):
    """
    使用重构后的流程运行视频问答推理，并包含超时重试和答案长度校验重试机制。

    :param client: 已经初始化好的 OpenAI 客户端实例。
    :param model_name: 使用的模型名称。
    :param video_path: 视频的路径（本地文件或URL）。
    :param question: 关于视频的问题。
    :param fps: 视频采样率。
    :param timeout: API 请求的超时时间（秒）。
    :param retries: 失败后的重试次数。
    :return: 模型回答的字符串。
    """
    print(f"\n[Thread] 正在处理视频: {Path(video_path).name}")

    try:
        # 如果是本地路径，转换为 file URI
        if os.path.exists(video_path) and not video_path.startswith("file://"):
            video_path = Path(video_path).resolve().as_uri()

        original_question = question
        
        # 外层循环：处理网络错误等导致的请求失败
        for attempt in range(retries):
            try:
                # 内层循环：处理答案超长问题，给模型最多10次机会修正
                for len_check_attempt in range(10):
                    # 在答案长度重试时，强化Prompt
                    if len_check_attempt > 0:
                        question = f"REMEMBER: Your answer MUST be 30 words or fewer. Question: {original_question}"
                    else:
                        question = original_question

                    prompt_text = f"<video>\nAnswer the question in 30 or fewer words.\n{question}"
                    
                    messages = [
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "video", 
                             "video": video_path, 
                             "fps": fps,
                            }
                        ]}
                    ]

                    # 2. 使用工具函数准备最终发送给 vLLM 的消息和参数
                    print(f'开始准备vllm的输入消息，视频路径：{video_path}, fps：{fps}')
                    start_time = time.time()
                    vllm_messages, video_kwargs = prepare_message_for_vllm(messages)
                    duration = time.time() - start_time
                    print(f'准备vllm的输入消息完成，耗时: {duration:.2f}s, 视频路径：{video_path}, fps：{fps}, video_kwargs: {video_kwargs}')

                    print(f'开始调用vllm的api, 视频路径：{video_path}, fps：{fps}')
                    start_time = time.time()
                    params = {
                        "model": model_name,
                        "messages": vllm_messages,
                        "max_tokens": 2048,
                        "temperature": 0.0,
                        "extra_body": {"mm_processor_kwargs": video_kwargs}
                    }

                    response = client.chat.completions.create(**params, timeout=timeout)
                    answer = response.choices[0].message.content
                    
                    duration = time.time() - start_time
                    print(f'调用vllm的api完成，耗时: {duration:.2f}s, 视频路径：{video_path}, fps：{fps}')

                    # 检查答案长度
                    if len(answer.split()) <= 30:
                        print(f"[Thread] 视频 {Path(video_path).name} 答案符合长度要求: {answer}")
                        return answer # 答案合格，直接返回

                    print(f"[Thread] 视频 {Path(video_path).name} 答案={answer}, 长度={len(answer.split())}, 超长 (第 {len_check_attempt + 1}/10 次尝试)，将重试...")
                
                # 如果10次尝试后答案仍然超长，就返回最后一次得到的长答案
                print(f"[Thread] 视频 {Path(video_path).name} 长度校验重试10次后仍失败，将使用最后一次结果。")
                return answer

            except Exception as e:
                print(f"[Thread] 视频 {Path(video_path).name} 请求失败，第 {attempt + 1}/{retries} 次网络重试: {e}")
                if attempt + 1 == retries:
                    print(f"[Thread] 视频 {Path(video_path).name} 所有网络重试均失败。")
                    return None # 网络重试失败，返回 None
                time.sleep(3)  # 在重试前等待

    except Exception as e:
        print(f"处理视频 {video_path} 时发生严重错误: {e}")
        return None # 内部处理失败，返回 None


def truncate_answer(answer_text: str, max_words: int = 30) -> str:
    """
    根据比赛要求，将答案截断到指定的最多单词数。
    """
    words = answer_text.split()
    if len(words) > max_words:
        return " ".join(words[:max_words])
    return answer_text


def flush_results_to_checkpoint(
    results_buffer: dict, buffer_lock: Lock, checkpoint_path: str
) -> int:
    """
    线程安全地将内存缓冲区中的所有结果写入检查点文件，并清空缓冲区。
    
    :return: 成功写入的条目数量。
    """
    with buffer_lock:
        if not results_buffer:
            return 0
        
        num_to_save = len(results_buffer)
        try:
            with open(checkpoint_path, 'a', encoding='utf-8') as f:
                for result in results_buffer.values():
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            
            results_buffer.clear()
            return num_to_save
        except IOError as e:
            print(f"错误: 写入检查点文件失败 -> {e}")
            return 0


def process_single_item(item, client, model_name, video_root_path, fps, results_buffer, buffer_lock):
    """
    处理单个问答条目的工作函数，并将结果存入内存缓冲区。
    """
    question_id = item.get("question_id")
    video_id = item.get("video")
    question_text = item.get("question")

    if not all([question_id, video_id, question_text]):
        print(f"警告: 跳过格式不完整的条目 -> {item}")
        return
    
    video_path = f"{video_root_path}/{video_id}.mp4"
    if not os.path.exists(video_path):
        print(f"警告: 视频文件不存在，跳过问题 ID {question_id} ({video_path})")
        return

    # 获取模型回答
    answer = run_inference_with_utils(
        client=client,
        model_name=model_name,
        video_path=video_path,
        question=question_text,
        fps=fps
    )

    # 如果推理失败 (返回 None)，则直接终止此任务
    if answer is None:
        print(f"[Thread] 任务 {question_id} 因推理失败而被跳过。")
        return

    # 根据比赛规则，对答案进行截断
    truncated_answer = truncate_answer(answer)

    # 准备结果字典
    result_to_save = {
        "question_id": question_id,
        "question": question_text,
        "answer": truncated_answer
    }
    
    # 将结果存入线程安全的缓冲区
    with buffer_lock:
        # 使用 question_id 和 video_id 组合作为唯一键
        key = f"{question_id}_{video_id}"
        results_buffer[key] = result_to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run video VQA evaluation with configurable FPS.")
    parser.add_argument(
        "--fps", 
        type=float, 
        default=2.0, 
        help="Frames per second to sample from the video. (default: 2.0)"
    )
    parser.add_argument(
        "--api_url",
        type=str,
        default="http://0.0.0.0:18903/v1",
        help="API URL for the model."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen25-VL-72B-Instruct",
        help="The name of the model to use."
    )
    parser.add_argument(
        "--video_root_path",
        type=str,
        default='./data/VR-Ads/videos',
        help="Root directory where videos are stored."
    )
    parser.add_argument(
        "--question_file_path",
        type=str,
        default='./data/VR-Ads/adsqa_question_file.json',
        help="Path to the question JSON file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../data/mars/VR-Ads/",
        help="Directory to save the output submission file."
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=32,
        help="Maximum number of threads for processing."
    )
    parser.add_argument(
        "--checkpoint_batch_size",
        type=int,
        default=8,
        help="Number of results to buffer before writing a checkpoint."
    )
    args = parser.parse_args()

    # --- 文件和路径配置 ---
    question_file_path = args.question_file_path
    # 让输出文件名包含 FPS 参数，便于区分不同实验的结果
    output_filename = f"submission_results_{args.model_name}_fps_{int(args.fps)}.json"
    output_file_path = os.path.join(args.output_dir, output_filename)
    # 使用 .jsonl 文件作为检查点，便于追加写入
    checkpoint_file_path = output_file_path + ".ckpt.jsonl"

    print(f"输出文件: {output_file_path}")
    print(f"检查点文件: {checkpoint_file_path}")

    # --- 推理参数 ---
    # MAX_WORKERS = 32  # 设置并发线程数
    # CHECKPOINT_BATCH_SIZE = 8 # 每处理完5条结果就保存一次检查点

    # 1. 初始化 OpenAI 客户端
    client = OpenAI(api_key=args.api_key, base_url=args.api_url)

    # --- 新增：用于在内存中缓存结果的线程安全结构 ---
    results_buffer = {}
    buffer_lock = Lock()

    # 2. 加载问题文件
    print(f"正在从 {question_file_path} 加载问题...")
    try:
        with open(question_file_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        print(f"成功加载 {len(questions_data)} 个问题。")
    except FileNotFoundError:
        print(f"错误: 问题文件未找到 -> {question_file_path}")
        exit()
    except json.JSONDecodeError:
        print(f"错误: JSON 文件格式无效 -> {question_file_path}")
        exit()

    # --- 断点续跑逻辑 ---
    processed_question_ids = set()
    if os.path.exists(checkpoint_file_path):
        try:
            with open(checkpoint_file_path, 'r', encoding='utf-8') as f:
                # 从检查点文件加载已处理的问题ID
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if 'question_id' in item:
                            processed_question_ids.add(item['question_id'])
            if processed_question_ids:
                print(f"检测到检查点文件，已处理 {len(processed_question_ids)} 个问题，将从断点处继续。")
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"警告: 检查点文件 {checkpoint_file_path} 读取失败或格式错误，将重新开始。")
            processed_question_ids = set()

    # 筛选出尚未处理的问题
    tasks_to_run = [item for item in questions_data if item.get('question_id') not in processed_question_ids]
    if not tasks_to_run:
        print("所有问题均已处理完毕。")
        exit()
    else:
        total_tasks = len(questions_data)
        completed_tasks = len(processed_question_ids)
        print(f"总任务数: {total_tasks}, 已完成: {completed_tasks}, 待处理: {len(tasks_to_run)}")
    # --------------------

    # 3. 使用线程池进行多线程处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 提交所有任务到线程池
        futures = {
            executor.submit(
                process_single_item, 
                item, client, args.model_name, args.video_root_path, args.fps, results_buffer, buffer_lock
            ) for item in tasks_to_run
        }

        print(f"\n已提交 {len(tasks_to_run)} 个任务到线程池...")

        # 使用 as_completed 实时处理完成的任务并显示进度条
        for future in tqdm(as_completed(futures), total=len(tasks_to_run), desc="多线程批量写入处理"):
            try:
                # 获取任务结果，如果任务执行中抛出异常，这里会重新抛出
                future.result()
            except Exception as e:
                # 记录或处理在 process_single_item 中发生的严重错误
                tqdm.write(f"\n[错误] 一个任务执行失败: {e}")

            # 检查缓冲区是否已满，满了就写入文件
            if len(results_buffer) >= args.checkpoint_batch_size:
                saved_count = flush_results_to_checkpoint(results_buffer, buffer_lock, checkpoint_file_path)
                if saved_count > 0:
                    # 使用 tqdm.write 避免破坏进度条显示
                    tqdm.write(f"[检查点] 缓存区已满，成功保存 {saved_count} 条结果到文件。")

    # 4. 结束 - 保存缓冲区中剩余的所有结果
    print("\n所有任务处理完毕。正在保存缓冲区中剩余的结果...")
    final_saved_count = flush_results_to_checkpoint(results_buffer, buffer_lock, checkpoint_file_path)
    if final_saved_count > 0:
        print(f"[检查点] 成功保存最后 {final_saved_count} 条结果。")

    # 5. 整合检查点文件到最终输出文件
    print("\n所有任务处理完毕。正在整合结果到最终文件...")
    final_results = []
    if os.path.exists(checkpoint_file_path):
        with open(checkpoint_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        final_results.append(json.loads(line))
                    except json.JSONDecodeError:
                        print(f"警告: 检查点文件中发现无效的JSON行，已跳过: {line}")

    # 写入最终的JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=4, ensure_ascii=False)

    print(f"任务完成。最终结果已保存至 '{output_file_path}'，共包含 {len(final_results)} 条记录。")
