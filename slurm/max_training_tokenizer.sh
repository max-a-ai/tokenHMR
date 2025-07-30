#!/bin/bash -l
#SBATCH --job-name=tokenizer
#SBATCH --output=/hnvme/workspace/v103fe17-tokenhmr/output/tokenizer_%j.txt
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h100:1
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

# Print script filename as header
JOB_NAME="tokenhmr-tokenizer"
SCRIPT_PATH="tokenHMR/slurm/max_training_tokenizer.sh"
echo -e "\n##########\n$SCRIPT_PATH\n##########\n"

# Activate Conda
module add python
conda activate tokenhmr

# find the data
STORAGE_DIR="$(ws_find tokenhmr)/data/dataset/tokenization_data"
WORK_DIR="$(ws_find tokenhmr)"

# find $STORAGE_DIR -type f -name '*.tar.gz' | xargs -P 8 -I{} bash -c 'mkdir -p $TMPDIR/tmp_{} && tar xzf {} -C $TMPDIR/tmp_{} && cp -r $TMPDIR/tmp_{}/* $TMPDIR'
find "$STORAGE_DIR" -type f -name '*.tar.gz' | xargs -P 8 -I{} tar -xzf {} -C "$TMPDIR"
set -x

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

cd $WORK_DIR/tokenHMR

export OMP_NUM_THREADS=32
export DATA_ROOT_OVERRIDE=$TMPDIR

# Create temporary config with modified DATA_ROOT
cp tokenization/configs/tokenizer_amass_moyo.yaml /tmp/tokenizer_temp.yaml
sed -i "s|DATA_ROOT: '.*'|DATA_ROOT: '$TMPDIR'|" /tmp/tokenizer_temp.yaml

python tokenization/train_poseVQ.py \
    --cfg /tmp/tokenizer_temp.yaml

# torchrun --nproc_per_node=1 \
#     --rdzv_endpoint=localhost:${PORT} \
#     train.py \
#     --cfg_file=$STORAGE_DIR/lif/max_config.yaml \
#     --set \
#         DATA_CONFIG.EXP_NAME $JOB_NAME \
#         DATA_CONFIG.ROOT_DIR $TMPDIR \
#         MODEL_CONFIG.DECODER_PRE $STORAGE_DIR/data/checkpoints/epoch=35-step=1000000.ckpt \
#         OPTIMIZATION_CONFIG.LOG_DIR $STORAGE_DIR/lif_eval/log \
#         OPTIMIZATION_CONFIG.SAVE_DIR $STORAGE_DIR/lif_eval/save \
#         OPTIMIZATION_CONFIG.DEBUG_DIR $STORAGE_DIR/lif_eval/debug \
#         SMPL.DATA_DIR $STORAGE_DIR/data/body_models \
#         SMPL.MODEL_PATH $STORAGE_DIR/data/body_models/smpl \
#         SMPL.JOINT_REGRESSOR_EXTRA $STORAGE_DIR/data/body_models/J_regressor_coco.npy \
#         SMPL.MEAN_PARAMS $STORAGE_DIR/data/body_models/smpl_mean_params.npz \
#         DATA_CONFIG.KP_TYPE 2D \
#         OPTIMIZATION_CONFIG.LOSS_TYPES "['2D']" \

# Deactivate the virtual environment at the end
conda deactivate