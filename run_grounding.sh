#!/bin/bash
# -----------------------------------------------------------------------------
# 运行 Visual Grounding (VG-RS) 任务的推理脚本
#
# 使用方法:
# 1. 根据您的环境修改下面的 "配置" 部分，特别是模型和数据路径。
# 2. 修改 INFERENCE_MODE 变量来选择 'hf' 或 'client' 模式。
# 3. 在终端中直接运行: bash run_grounding.sh
# -----------------------------------------------------------------------------

# --- 配置 ---
# 推理模式, 可选 'hf' (Hugging Face) 或 'client' (VLLM API)
INFERENCE_MODE="hf"

# --- 通用配置 ---
TASK="grounding"
JSON_PATH="./data/VG-RS/VG-RS-question.json"


IMAGE_BASE_DIR="./data/VG-RS"
MODEL_PATH="Zach996/ActiveAlphaAgent-VG-RS" # https://huggingface.co/Zach996/ActiveAlphaAgent-VG-RS


OUTPUT_DIR="./data/VG-RS/result/submission_qwen25_vl_distill_7b_${INFERENCE_MODE}"
MODEL_NAME="Qwen25-7B-Distill-VG"
GPU_IDS="0,1,2,3"

# --- Client 模式特定配置 ---
VLLM_PORT="18901" # 为 grounding 服务分配一个端口
NUM_WORKERS=16

# --- 脚本执行 ---

# 根据 GPU_IDS 计算 tensor-parallel-size
IFS=',' read -ra GPUS <<< "$GPU_IDS"
TP_SIZE=${#GPUS[@]}

# 创建输出目录
echo "创建输出目录: $OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

if [ "$INFERENCE_MODE" == "hf" ]; then
    # ------------------- HF 模式 -------------------
    echo "使用 HF 模式进行推理..."
    python3 eval_grounding_vqa.py \
        --task $TASK \
        --json_path $JSON_PATH \
        --image_base_dir $IMAGE_BASE_DIR \
        --output_dir "$OUTPUT_DIR" \
        --model_name $MODEL_NAME \
        --inference_mode hf \
        --model_path $MODEL_PATH \
        --gpu_ids "$GPU_IDS"

elif [ "$INFERENCE_MODE" == "client" ]; then
    # ------------------- CLIENT 模式 -------------------
    echo "使用 CLIENT 模式进行推理..."
    
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
        VLLM_LOG_FILE="vllm_server_grounding_${VLLM_PORT}.log"
        echo "GPU IDs: $GPU_IDS, TP_SIZE: $TP_SIZE"
        echo "VLLM 服务日志将保存在: $VLLM_LOG_FILE"

        CUDA_VISIBLE_DEVICES=$GPU_IDS vllm serve $MODEL_PATH \
            --port $VLLM_PORT \
            --served-model-name $MODEL_NAME \
            --gpu-memory-utilization 0.80 \
            --max-model-len 20480 \
            --tensor-parallel-size $TP_SIZE > "$VLLM_LOG_FILE" 2>&1 &
        
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
    echo "开始 Visual Grounding 推理 (client)..."
    python3 eval_grounding_vqa.py \
        --task $TASK \
        --json_path $JSON_PATH \
        --image_base_dir $IMAGE_BASE_DIR \
        --output_dir "$OUTPUT_DIR" \
        --model_name $MODEL_NAME \
        --inference_mode client \
        --port $VLLM_PORT \
        --num_workers $NUM_WORKERS

    # 4. 提示用户手动停止服务
    if [ "$USE_EXISTING_SERVER" = false ] && [ -n "$VLLM_PID" ]; then
        echo "-------------------------------------"
        echo "推理完成!"
        echo "重要提示: VLLM 服务仍在后台运行。"
        echo "请使用以下命令停止由本脚本启动的服务: kill $VLLM_PID"
        echo "-------------------------------------"
    else
        echo "-------------------------------------"
        echo "推理完成!"
        echo "本次任务使用了已存在的服务，未启动新服务。"
        echo "-------------------------------------"
    fi

else
    echo "错误: 无效的 INFERENCE_MODE: '$INFERENCE_MODE'. 请选择 'hf' 或 'client'."
    exit 1
fi

echo "-------------------------------------"
echo "任务完成!"
echo "结果已保存到: $OUTPUT_DIR"
echo "-------------------------------------"
