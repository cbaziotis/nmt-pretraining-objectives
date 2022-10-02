#!/bin/bash

DATASET=$1
#DATASET=flores_neen

# main paths
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
  L1=si
  L2=en

  SPM=$DATA_PATH/spm/sien/mono.sien.20000
  OUT_SPM=$DATA_PATH/flores/parallel_sien_spm
  OUT_BIN=$DATA_PATH/flores/parallel_sien_bin

  TRAIN_RAW=$DATA_PATH/flores/wiki_si_en_bpe5000/train
  VALID_RAW=$DATA_PATH/flores/wiki_si_en_bpe5000/valid
  TEST_RAW=$DATA_PATH/flores/wiki_si_en_bpe5000/test

elif [ "$DATASET" == "neen" ]; then
  L1=ne
  L2=en
  SPM=$DATA_PATH/spm/neen/mono.neen.20000
  OUT_SPM=$DATA_PATH/flores/parallel_neen_spm
  OUT_BIN=$DATA_PATH/flores/parallel_neen_bin

  TRAIN_RAW=$DATA_PATH/flores/wiki_ne_en_bpe5000/train
  VALID_RAW=$DATA_PATH/flores/wiki_ne_en_bpe5000/valid
  TEST_RAW=$DATA_PATH/flores/wiki_ne_en_bpe5000/test

elif [ "$DATASET" == "deen" ]; then
  L1=de
  L2=en
  SPM=$DATA_PATH/spm/deen/mono.deen.60000
  OUT_SPM=$DATA_PATH/deen/parallel_spm
  OUT_BIN=$DATA_PATH/deen/parallel_bin

  TRAIN_RAW=$DATA_PATH/deen/parallel/news-commentary-v13.de-en
  VALID_RAW=$DATA_PATH/deen/parallel/newstest2018-deen
  TEST_RAW=$DATA_PATH/deen/parallel/newstest2019-deen

else
  echo "unknown dataset"
fi

#---------------------------------------------------------------------------
# 2. Use the SPM from monolingual data to tokenize the parallel data
#---------------------------------------------------------------------------
mkdir -p "$OUT_SPM"

echo "Tokenizing the parallel data using the pretrained SPM:'$SPM'"
for LANG in $L1 $L2; do
  "$SPM_SRC/spm_encode" --model=$SPM.model --output_format=piece <$TRAIN_RAW.$LANG >$OUT_SPM/train.$LANG
  "$SPM_SRC/spm_encode" --model=$SPM.model --output_format=piece <$VALID_RAW.$LANG >$OUT_SPM/valid.$LANG
  "$SPM_SRC/spm_encode" --model=$SPM.model --output_format=piece <$TEST_RAW.$LANG >$OUT_SPM/test.$LANG
done

#---------------------------------------------------------------------------
# 3. Binarize the tokenized parallel data
#---------------------------------------------------------------------------
fairseq-preprocess \
  --trainpref $OUT_SPM/train \
  --validpref $OUT_SPM/valid \
  --testpref $OUT_SPM/test \
  --destdir $OUT_BIN \
  --srcdict $SPM.dict.txt \
  --tgtdict $SPM.dict.txt \
  --source-lang $L1 \
  --target-lang $L2 \
  --bpe sentencepiece \
  --workers "$(nproc)"

cp "$SPM.dict.txt" "$OUT_BIN/dict.txt"
cp $SPM.model $OUT_BIN/spm.model
cp $SPM.vocab $OUT_BIN/spm.vocab
