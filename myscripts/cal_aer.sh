#sourc language
src=de
#target language
tgt=en
CODE_DIR=workspace/cross-align_final/
python=workspace/bin/python
scripts=$CODE_DIR/alignment-tools
#gold data
GOLD_FILE=$CODE_DIR/data/${src}${tgt}.talp
#outputs path
OUTPUT_FILE=$CODE_DIR/data/

export PYTHONPATH=$CODE_DIR:$PYTHONPATH

$python $scripts/aer.py $GOLD_FILE $OUTPUT_FILE/${src}${tgt}.out  --fAlpha 0.5 --oneRef
#$python $scripts/aer.py $GOLD_FILE $OUTPUT_FILE/${src}${tgt}.out  --fAlpha 0.5
