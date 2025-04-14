#download the data from https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data (it requires creating an account on Kaggle)
# echo "make sure you have downloaded the data from https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data (all_data.csv) and placed it in data/toxicity/jigsaw-unintended-bias-in-toxicity-classification"

DATADIR=$1
# data/toxicity/jigsaw-unintended-bias-in-toxicity-classification
#processing the data
echo "preprocessing the data"
python -u data/toxicity/create_jigsaw_toxicity_data.py $DATADIR

# 检查文件是否存在
if [ ! -f "${DATADIR}/toxicity_eq0.jsonl" ]; then
    echo "错误：文件 ${DATADIR}/toxicity_eq0.jsonl 不存在"
    exit 1
fi

if [ ! -f "${DATADIR}/toxicity_gte0.5.jsonl" ]; then
    echo "错误：文件 ${DATADIR}/toxicity_gte0.5.jsonl 不存在"
    exit 1
fi
# 输出${DATADIR}
echo "DATADIR: $DATADIR"
# 获取文件行数
# N=$(wc -l ${DATADIR}/toxicity_eq0.jsonl | cut -d ' ' -f1)
# n=$(wc -l ${DATADIR}/toxicity_gte0.5.jsonl | cut -d ' ' -f1)
N=$(cat ${DATADIR}/toxicity_eq0.jsonl | wc -l | tr -d ' ')
n=$(cat ${DATADIR}/toxicity_gte0.5.jsonl | wc -l | tr -d ' ')

# 输出N和n
echo "N: $N"
echo "n: $n"

# 检查是否成功获取到行数
if [ -z "$N" ] || [ -z "$n" ]; then
    echo "错误：无法获取文件行数"
    exit 1
fi
python -u data/toxicity/random_sample.py ${DATADIR}/toxicity_eq0.jsonl ${DATADIR}/toxicity_eq0_subsample.jsonl $N $n


TEST=2000
DEV=2000
TRAIN=$(($n-$DEV-$TEST))
python -u data/toxicity/split_train_dev_test.py ${DATADIR}/toxicity_eq0_subsample.jsonl 0 $TRAIN $DEV $TEST $DATADIR
python -u data/toxicity/split_train_dev_test.py ${DATADIR}/toxicity_gte0.5.jsonl 1 $TRAIN $DEV $TEST $DATADIR
