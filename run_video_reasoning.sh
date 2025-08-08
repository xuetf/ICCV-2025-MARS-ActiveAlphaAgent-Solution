#!/bin/bash
# -----------------------------------------------------------------------------
# 运行 Video Question Answering (VR-Ads) 任务的脚本
#
# 使用方法:
# 1. 根据您的环境修改下面的 "配置" 部分。
# 2. 脚本会先检查服务是否存在，如果不存在则启动服务，然后运行推理客户端。
# 3. 运行结束后，如果脚本启动了服务，会提示您手动停止。
# 4. 在终端中直接运行: bash run_video_vqa.sh
# -----------------------------------------------------------------------------

# --- 配置 ---
# --- VLLM 服务配置 ---
VLLM_PORT="18903"
MODEL_PATH="Qwen/Qwen2.5-VL-72B-Instruct"
GPU_IDS="0,1,2,3,4,5,6,7"
TP_SIZE=8
MODEL_NAME="Qwen25-VL-72B-Instruct"
API_URL="http://0.0.0.0:${VLLM_PORT}/v1"



# --- 推理客户端配置 ---
VIDEO_ROOT_PATH="./data/VR-Ads/videos"
QUESTION_FILE_PATH="./data/VR-Ads/adsqa_question_file.json"
OUTPUT_DIR="./data/VR-Ads/result/submission_qwen25_vl_72b_instruct_fps_2"
FPS=2.0
MAX_WORKERS=16

# --- 脚本执行 ---

VLLM_PID=""
USE_EXISTING_SERVER=false

# 检查现有服务
echo "正在检查端口 $VLLM_PORT 是否已有服务并提供模型 '$MODEL_NAME'..."
if curl -s "http://0.0.0.0:${VLLM_PORT}/v1/models" | grep -q "\"id\":\"$MODEL_NAME\""; then
    echo "检测到服务已在运行并提供所需模型。将使用现有服务。"
    USE_EXISTING_SERVER=true
else
    echo "未检测到可用服务，或模型不匹配。将启动新的 VLLM 服务。"
    # 1. 在后台启动 VLLM 服务
    VLLM_LOG_FILE="vllm_server_video_${VLLM_PORT}.log"
    echo "GPU IDs: $GPU_IDS, TP_SIZE: $TP_SIZE"
    echo "VLLM 服务日志将保存在: $VLLM_LOG_FILE"

    CUDA_VISIBLE_DEVICES=$GPU_IDS vllm serve $MODEL_PATH \
        --port $VLLM_PORT \
        --gpu-memory-utilization 0.80 \
        --max-model-len 128000 \
        --served-model-name $MODEL_NAME \
        --tensor-parallel-size $TP_SIZE \
        --limit-mm-per-prompt image=0,video=1 > "$VLLM_LOG_FILE" 2>&1 &

    VLLM_PID=$!
    echo "VLLM 服务已在后台启动，PID: $VLLM_PID"

    # 2. 轮询检查 VLLM 服务状态
    echo "正在等待 VLLM 服务启动..."
    MAX_RETRIES=60
    RETRY_INTERVAL=20
    HEALTH_CHECK_URL="http://0.0.0.0:${VLLM_PORT}/health"

    is_ready=false
    for i in $(seq 1 $MAX_RETRIES); do
        if curl -s -f "$HEALTH_CHECK_URL" > /dev/null; then
            echo "VLLM 服务已就绪！"
            is_ready=true
            break
        fi
        echo "等待 VLLM 服务... (尝试 $i/$MAX_RETRIES)"
        sleep $RETRY_INTERVAL
    done

    if [ "$is_ready" = false ]; then
        echo "错误: VLLM 服务在 $(($MAX_RETRIES * $RETRY_INTERVAL)) 秒内未能启动。"
        echo "请检查日志文件: $VLLM_LOG_FILE"
        if [ -n "$VLLM_PID" ]; then
            echo "正在停止后台服务进程 (PID: $VLLM_PID)..."
            kill $VLLM_PID
        fi
        exit 1
    fi
fi

# 3. 运行推理客户端
echo "开始 Video VQA 推理..."
mkdir -p $OUTPUT_DIR

python3 eval_video_reasoning.py \
    --api_url "$API_URL" \
    --model_name "$MODEL_NAME" \
    --video_root_path "$VIDEO_ROOT_PATH" \
    --question_file_path "$QUESTION_FILE_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --fps $FPS \
    --max_workers $MAX_WORKERS

# 4. 提示用户手动停止服务
if [ "$USE_EXISTING_SERVER" = false ] && [ -n "$VLLM_PID" ]; then
    echo "-------------------------------------"
    echo "Video VQA 推理完成!"
    echo "结果已保存到: $OUTPUT_DIR"
    echo "重要提示: VLLM 服务仍在后台运行。"
    echo "请使用以下命令停止由本脚本启动的服务: kill $VLLM_PID"
    echo "-------------------------------------"
else
    echo "-------------------------------------"
    echo "Video VQA 推理完成!"
    echo "结果已保存到: $OUTPUT_DIR"
    echo "本次任务使用了已存在的服务，未启动新服务。"
    echo "-------------------------------------"
fi
