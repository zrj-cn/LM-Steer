# ./experiments/pipeline.sh {第一个参数（必须）} {第二个参数（可选）}
# 第一个参数说明：
# 0：执行第0部分（准备数据）
# 1：执行第1部分（训练）
# 2：执行第2部分（生成）
# 3：执行第3部分（评估）
# -1：执行所有部分
# 第二个参数说明：
# （可选）针对第二部分代码，指定生成的样本数量，如果不指定则使用原始的10k样本文件
# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "请提供执行部分的第一个参数（0、1、2、3、4或-1）"
    exit 1
fi

PART=$1
TRIAN_MODEL=gpt2-large
# ======detoxicity parameters=============
TRIAL=detoxification-$TRIAN_MODEL
# ======sentiment parameters=============
source=positive
control=-5
# ========================================================
# 执行第0部分，创建logs文件夹
if [ $PART -eq 0 ] || [ $PART -eq -1 ]; then
    echo "执行第0部分..."
    mkdir -p logs/$TRIAL
fi

# ========================================================
# 执行第1部分，训练模型
if [ $PART -eq 1 ] || [ $PART -eq -1 ]; then
    echo "执行第1部分，训练模型..."
    PYTHONPATH=. python experiments/training/train.py \
        --dataset_name toxicity \
        --data_dir data/toxicity/jigsaw-unintended-bias-in-toxicity-classification \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $TRIAN_MODEL --cuda \
        --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
        --batch_size 32 --max_length 256 \
        --n_steps 1000 --lr 1e-2
fi

# ========================================================
# 执行第2部分，生成
if [ $PART -eq 2 ] || [ $PART -eq -1 ]; then
    echo "执行第2部分，生成语句..."
    
    # 获取第二个参数（样本数量）
    NUM_SAMPLES=$2
    
    # 如果指定了样本数量
    if [ ! -z "$NUM_SAMPLES" ]; then
        echo "生成 $NUM_SAMPLES 个样本..."
        # 执行采样脚本
        PYTHONPATH=. python data/prompts/sample_prompts.py \
            --num_samples $NUM_SAMPLES \
            --input_file data/prompts/nontoxic_prompts-10k.jsonl \
            --output_file data/prompts/nontoxic_prompts/nontoxic_prompts-$NUM_SAMPLES.jsonl
        
        # 使用生成的样本文件进行生成
        PYTHONPATH=. python experiments/training/generate.py \
            --eval_file data/prompts/nontoxic_prompts/nontoxic_prompts-$NUM_SAMPLES.jsonl \
            --output_file logs/$TRIAL/predictions-$NUM_SAMPLES.jsonl \
            --ckpt_name logs/$TRIAL/checkpoint.pt \
            --model $TRIAN_MODEL --cuda \
            --adaptor_class multiply --num_steers 2 --rank 1000 \
            --max_length 256 --verbose --steer_values 5 1
    else
        # 若未指定采样，使用原始的10k样本文件
        PYTHONPATH=. python experiments/training/generate.py \
            --eval_file data/prompts/nontoxic_prompts-10k.jsonl \
            --output_file logs/$TRIAL/predictions.jsonl \
            --ckpt_name logs/$TRIAL/checkpoint.pt \
            --model $TRIAN_MODEL --cuda \
            --adaptor_class multiply --num_steers 2 --rank 1000 \
            --max_length 256 --verbose --steer_values 5 1
    fi
fi

# ========================================================
# 执行第3部分，评估
if [ $PART -eq 3 ] || [ $PART -eq -1 ]; then
    echo "执行第3部分，评估效果..."
    
    # 获取第二个参数（样本数量）
    NUM_SAMPLES=$2
    
    # 根据是否指定样本数量选择对应的预测文件和结果文件
    if [ ! -z "$NUM_SAMPLES" ]; then
        PRED_FILE="logs/$TRIAL/predictions-$NUM_SAMPLES.jsonl"
        RESULT_FILE="result_stats-$NUM_SAMPLES.txt"
    else
        PRED_FILE="logs/$TRIAL/predictions.jsonl"
        RESULT_FILE="result_stats.txt"
    fi
    
    python experiments/evaluation/evaluate2.py \
        --generations_file $PRED_FILE \
        --metrics toxicity,ppl-big,dist-n \
        --output_file $RESULT_FILE
    echo "Detoxification results:"
    cat logs/$TRIAL/$RESULT_FILE
fi

# ================sentiment part==========================
# ========================================================
# 执行第4部分，训练模型
if [ $PART -eq 4 ] || [ $PART -eq -1 ]; then
    PYTHONPATH=. python experiments/training/train.py \
        --dataset_name sentiment-sst5 \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model $TRIAN_MODEL --cuda \
        --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
        --batch_size 32 --max_length 256 \
        --n_steps 1000 --lr 1e-2 --regularization 1e-6 --epsilon 1e-3
fi