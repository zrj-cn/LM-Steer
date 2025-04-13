# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "请提供执行部分的参数（0,1,2 或 -1）"
    exit 1
fi

PART=$1
TRIAL=detoxification-gpt2-large

# 执行第0部分
if [ $PART -eq 0 ] || [ $PART -eq -1 ]; then
    echo "执行第0部分..."
    mkdir -p logs/$TRIAL
fi

# 执行第1部分
if [ $PART -eq 1 ] || [ $PART -eq -1 ]; then
    echo "执行第1部分..."
    PYTHONPATH=. python experiments/training/train.py \
        --dataset_name toxicity \
        --data_dir data/toxicity/jigsaw-unintended-bias-in-toxicity-classification \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model gpt2-large --cuda \
        --adaptor_class multiply --num_steers 2 --dummy_steer 1 --rank 1000 \
        --batch_size 32 --max_length 256 \
        --n_steps 1000 --lr 1e-2
fi

# 执行第2部分
if [ $PART -eq 2 ] || [ $PART -eq -1 ]; then
    echo "执行第2部分..."
    PYTHONPATH=. python experiments/training/generate.py \
        --eval_file data/prompts/nontoxic_prompts-10k.jsonl \
        --output_file logs/$TRIAL/predictions.jsonl \
        --ckpt_name logs/$TRIAL/checkpoint.pt \
        --model gpt2-large --cuda \
        --adaptor_class multiply --num_steers 2 --rank 1000 \
        --max_length 256 --verbose --steer_values 5 1
fi