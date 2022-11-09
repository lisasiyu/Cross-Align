#source language
src=ro
#target language
tgt=en
CODE_DIR=/apdcephfs_cq2/share_47076/lisalai/code/WordAlignment/cross-align
#training data
TRAIN_FILE=/apdcephfs/share_47076/lisalai/code/WordAlignment/data/roen/orig/train.${src}-${tgt}
#pre-trained language model
MODEL_NAME_OR_PATH=/apdcephfs/share_47076/lisalai/code/huggingface/bert-base-multilingual-cased
#output of stage1 model
OUTPUT_DIR1=$CODE_DIR/outputs/stage1.${src}_${tgt}
#python path
python=/apdcephfs/share_47076/lisalai/anaconda3/envs/cross-align/bin/python

N_GPU=`nvidia-smi -L | wc -l`

if [ ! -d $OUTPUT_DIR1 ];then
    mkdir -p $OUTPUT_DIR1
    chmod -R 777 $OUTPUT_DIR1
fi

echo "Start training stage1"
cp $CODE_DIR/src/transformers/models/bert/modeling_bert_stage1.py $CODE_DIR/src/transformers/models/bert/modeling_bert.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $python -u $CODE_DIR/run_train.py \
    --output_dir=$OUTPUT_DIR1 \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --cache_dir=$MODEL_NAME_OR_PATH \
    --extraction 'softmax' \
    --do_train \
    --train_tlm \
    --align_layer 12 \
    --alpha 0.2 \
    --self_m 10 \
    --cross_k 2 \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 12 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --learning_rate 5e-4 \
    --logging_steps 50 \
    --save_steps 100 \
    --max_steps 200000 \
    --seed 42 \
    --cache_data \
    --overwrite_output_dir

