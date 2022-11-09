#sourc language
src=de
#target language
tgt=en
CODE_DIR=workspace/cross-align
python=workspace/bin/python
#Cross-Align model path
MODEL_NAME_OR_PATH=workspace/cross-align-models/${src}${tgt}
#test data
DATA_FILE=$CODE_DIR/data/test.${src}-${tgt}
#outputs path
OUTPUT_FILE=$CODE_DIR/data/

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


