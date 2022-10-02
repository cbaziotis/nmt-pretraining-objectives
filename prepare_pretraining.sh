#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# This script contains the preprocessing pipeline for some predefined datasets.
# 1. It learns a joint sentencepiece model on the training data
# 2. It tokenizes with the sentencepice model all the data
# 3. It binarizes them for training with faireq
#-----------------------------------------------------------------------------------------------------------------------

DATASET=$1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CURRENT_PATH=$(readlink -f "$DIR")

bash "$CURRENT_PATH/install-tools.sh"
SPM_SRC=$CURRENT_PATH/tools/sentencepiece/build/src

DATA_PATH=$CURRENT_PATH/data
mkdir -p "$DATA_PATH"

#---------------------------------------------------------------------------
# define dataset fname paths
#---------------------------------------------------------------------------
if [ "$DATASET" == "sien" ]; then
  VOCAB_SIZE=20000
  L1=si
  L2=en
  TRAIN_RAW_L1=$DATA_PATH/flores/mono/mono.norm.sample5000000.$L1
  TRAIN_RAW_L2=$DATA_PATH/flores/mono/mono.sample5000000.$L2
  VALID_RAW_L2=$DATA_PATH/flores/wiki_si_en_bpe5000/valid.$L2
  VALID_RAW_L1=$DATA_PATH/flores/wiki_si_en_bpe5000/valid.$L1
  SPM_PATH=$DATA_PATH/spm/sien
  OUT_SPM=$DATA_PATH/flores/mono_sien_spm
  OUT_BIN=$DATA_PATH/flores/mono_sien_bin

elif [ "$DATASET" == "neen" ]; then
  VOCAB_SIZE=20000
  L1=ne
  L2=en
  TRAIN_RAW_L1=$DATA_PATH/flores/mono/mono.norm.sample5000000.$L1
  TRAIN_RAW_L2=$DATA_PATH/flores/mono/mono.sample5000000.$L2
  VALID_RAW_L2=$DATA_PATH/flores/wiki_ne_en_bpe5000/valid.$L2
  VALID_RAW_L1=$DATA_PATH/flores/wiki_ne_en_bpe5000/valid.$L1
  SPM_PATH=$DATA_PATH/spm/neen
  OUT_SPM=$DATA_PATH/flores/mono_neen_spm
  OUT_BIN=$DATA_PATH/flores/mono_neen_bin

elif [ "$DATASET" == "deen" ]; then
  VOCAB_SIZE=60000
  L1=de
  L2=en
  TRAIN_RAW_L1=$DATA_PATH/deen/mono/news_crawl.$L1
  TRAIN_RAW_L2=$DATA_PATH/deen/mono/news_crawl.$L2
  VALID_RAW_L1=$DATA_PATH/deen/parallel/newstest2018-deen.$L1
  VALID_RAW_L2=$DATA_PATH/deen/parallel/newstest2018-deen.$L2
  SPM_PATH=$DATA_PATH/spm/deen
  OUT_SPM=$DATA_PATH/deen/mono_spm
  OUT_BIN=$DATA_PATH/deen/mono_bin

else
  echo "unknown dataset key"
fi

mkdir -p $SPM_PATH
mkdir -p $OUT_SPM
mkdir -p $OUT_BIN
SPM=$SPM_PATH/mono.$DATASET.$VOCAB_SIZE

#---------------------------------------------------------------------------
# 1. Train the sentencepiece model (SPM)
#---------------------------------------------------------------------------
SPM_COVERAGE=0.9995

if [ ! -f "$SPM.model" ]; then
  echo "Training SPM..."
  "$SPM_SRC/spm_train" --input="${TRAIN_RAW_L1},${TRAIN_RAW_L2}" \
    --vocab_size=$VOCAB_SIZE \
    --character_coverage=$SPM_COVERAGE \
    --max_sentence_length=256 \
    --model_prefix=$SPM \
    --model_type=unigram \
    --input_sentence_size=100000 --shuffle_input_sentence=true

  # convert SPM vocab to fairseq dict for later use
  cut -f1 $SPM.vocab | tail -n +4 | sed "s/$/ 100/g" >$SPM.dict.txt

else
  echo "Using pretrained $SPM.model!"
fi

#---------------------------------------------------------------------------
# 2. Use the SPM to tokenize the data
#---------------------------------------------------------------------------
echo "Tokenizing..."
"$SPM_SRC/spm_encode" --model=$SPM.model --output_format=piece <$TRAIN_RAW_L1 | head -n 10000 >$OUT_SPM/train.$L1
"$SPM_SRC/spm_encode" --model=$SPM.model --output_format=piece <$TRAIN_RAW_L2 | head -n 10000 >$OUT_SPM/train.$L2
"$SPM_SRC/spm_encode" --model=$SPM.model --output_format=piece <$VALID_RAW_L1 >$OUT_SPM/valid.$L1
"$SPM_SRC/spm_encode" --model=$SPM.model --output_format=piece <$VALID_RAW_L2 >$OUT_SPM/valid.$L2

#---------------------------------------------------------------------------
# 3. Binarize the tokenized data
#---------------------------------------------------------------------------
echo "Binarizing..."

fairseq-preprocess --only-source \
  --trainpref $OUT_SPM/train.$L1 \
  --validpref $OUT_SPM/valid.$L1 \
  --destdir $OUT_BIN/$L1 \
  --srcdict $SPM.dict.txt \
  --bpe sentencepiece \
  --workers "$(nproc)"

fairseq-preprocess --only-source \
  --trainpref $OUT_SPM/train.$L2 \
  --validpref $OUT_SPM/valid.$L2 \
  --destdir $OUT_BIN/$L2 \
  --srcdict $SPM.dict.txt \
  --bpe sentencepiece \
  --workers "$(nproc)"

cp "$SPM.dict.txt" "$OUT_BIN/dict.txt"
cp $SPM.model $OUT_BIN/spm.model
cp $SPM.vocab $OUT_BIN/spm.vocab
