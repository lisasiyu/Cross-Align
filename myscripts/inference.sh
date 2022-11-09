#sourc language
src=de
#target language
tgt=en
CODE_DIR=/apdcephfs_cq2/share_47076/lisalai/code/WordAlignment/cross-align
python=/apdcephfs/share_47076/lisalai/anaconda3/envs/cross-align/bin/python
#Cross-Align model path
MODEL_NAME_OR_PATH=/apdcephfs_cq2/share_47076/lisalai/code/WordAlignment/cross-align-final-models/deen
#test data
DATA_FILE=$CODE_DIR/data/test.${src}-${tgt}
#outputs path
OUTPUT_FILE=$CODE_DIR/data/

scripts=$CODE_DIR/alignment-tools
#gold data
GOLD_FILE=$CODE_DIR/data/${src}${tgt}.talp




export PYTHONPATH=$CODE_DIR:$PYTHONPATH

cp $CODE_DIR/src/transformers/models/bert/modeling_bert_stage2.py $CODE_DIR/src/transformers/models/bert/modeling_bert.py

CUDA_VISIBLE_DEVICES=0
$python $CODE_DIR/run_align.py \
    --output_file=$OUTPUT_FILE/${src}${tgt}.out \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --cache_dir $MODEL_NAME_OR_PATH \
    --data_file=$DATA_FILE \
    --extraction 'softmax' \
    --softmax_threshold 0.15 \
    --beta 0.5 \
    --align_layer 11 \
    --batch_size 32
$python $scripts/aer.py $GOLD_FILE $OUTPUT_FILE/${src}${tgt}.out  --fAlpha 0.5 --oneRef

