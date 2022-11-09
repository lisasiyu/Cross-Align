#source language
src=de
#target language
tgt=en
CODE_DIR=workspace/cross-align
#training data
TRAIN_FILE=workspace/cross-align/train.${src}-${tgt}

#output of stage1 model
OUTPUT_DIR1=$CODE_DIR/outputs/stage1.${src}_${tgt}/
#ouput of stage2 model
OUTPUT_DIR2=$CODE_DIR/outputs/stage2.${src}_${tgt}
#python path
python=/bin/python

if [ ! -d $OUTPUT_DIR2 ];then
    mkdir -p $OUTPUT_DIR2
    chmod -R 777 $OUTPUT_DIR2
fi

echo "Start training stage2"
cp $CODE_DIR/src/transformers/models/bert/modeling_bert_stage2.py $CODE_DIR/src/transformers/models/bert/modeling_bert.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $python -u $CODE_DIR/run_train_stage2.py \
    --output_dir=$OUTPUT_DIR2 \
    --model_name_or_path=$OUTPUT_DIR1 \
    --stage1_model_name_or_path=$OUTPUT_DIR1 \
    --cache_dir=$OUTPUT_DIR1 \
    --extraction 'softmax' \
    --do_train \
    --freeze \
    --train_so \
    --align_layer 11 \
    --alpha 0.2 \
    --self_m 10 \
    --cross_k 2 \
    --train_data_file=$TRAIN_FILE \
    --per_gpu_train_batch_size 12 \
    --per_gpu_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --logging_steps 50 \
    --save_steps 500 \
    --max_steps 5000 \
    --cache_data \
    --overwrite_output_dir
