if [ $# -eq 0 ]; then
    echo "请提供执行部分的第一个参数（0、1、2、3、4或-1）"
    exit 1
fi

PART=$1
TRIAN_MODEL=gpt2-large
# ======detoxicity parameters=============
TRIAL=input-detoxification-$TRIAN_MODEL
# ======sentiment parameters=============
source=positive
control=-5

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
        --n_steps 1000 --lr 1e-2 \
        --adapted_component input_embedding
fi