#!/bin/bash

DATASET=$1

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CURRENT_PATH=$(readlink -f "$DIR")

DATA_PATH=$CURRENT_PATH/data
mkdir -p "$DATA_PATH"

#---------------------------------------------------------------------------
# define dataset fname paths
#---------------------------------------------------------------------------
if [ "$DATASET" == "sien" ]; then
  L1=si
  L2=en
  SPM_PREFIX=$DATA_PATH/spm/sien/mono.sien.20000
  MONO_PATH=$DATA_PATH/flores/mono_sien_spm
  PARA_PATH=$DATA_PATH/flores/parallel_sien_spm
  OUT_BIN=$DATA_PATH/flores/mono_sien_unmt_bin

elif [ "$DATASET" == "neen" ]; then
  L1=ne
  L2=en
  SPM_PREFIX=$DATA_PATH/spm/neen/mono.neen.20000
  MONO_PATH=$DATA_PATH/flores/mono_neen_spm
  PARA_PATH=$DATA_PATH/flores/parallel_neen_spm
  OUT_BIN=$DATA_PATH/flores/mono_neen_unmt_bin

elif [ "$DATASET" == "deen" ]; then
  L1=de
  L2=en
  SPM_PREFIX=$DATA_PATH/spm/deen/mono.deen.60000
  MONO_PATH=$DATA_PATH/deen/mono_spm
  PARA_PATH=$DATA_PATH/deen/parallel_spm
  OUT_BIN=$DATA_PATH/deen/mono_unmt_bin

else
  echo "unknown dataset"
fi

#---------------------------------------------------------------------------
# Binarize the tokenized data
#---------------------------------------------------------------------------
echo "Binarizing..."

binarize_parallel() {
  echo "Binarizing parallel data $1-$2..."

  # we use a smaller sub-sample of the validation set
  head -200 $PARA_PATH/valid.$1 >$PARA_PATH/valid.h100.$1
  head -200 $PARA_PATH/valid.$2 >$PARA_PATH/valid.h100.$2

  fairseq-preprocess \
    --source-lang $1 \
    --target-lang $2 \
    --validpref $PARA_PATH/valid.h100 \
    --destdir $OUT_BIN \
    --srcdict $SPM_PREFIX.dict.txt \
    --tgtdict $SPM_PREFIX.dict.txt \
    --workers "$(nproc)"

  rm $PARA_PATH/valid.h100.$1
  rm $PARA_PATH/valid.h100.$2
}

binarize_monolingual() {
  echo "Binarizing (tokenized) monolingual data $OUT_BIN..."
  fairseq-preprocess \
    --only-source \
    --trainpref $MONO_PATH/train \
    --validpref $MONO_PATH/valid \
    --destdir $OUT_BIN \
    --srcdict $SPM_PREFIX.dict.txt \
    --source-lang $1 \
    --bpe sentencepiece \
    --workers "$(nproc)"

  # create a frequency list wit a random subset of each language's
  # monolingual corpus, which will be used for masking the backtranslations
  STATF=$MONO_PATH/train.$1
  echo "Computing token statistics on '$STATF'..."
  python $CURRENT_PATH/infer_token_mask.py -fi $STATF -fo $OUT_BIN/tok_probs.$1 -vocab $SPM_PREFIX.vocab
  #  shuf -n 5000000 $STATF | tr ' ' '\n' | sort | uniq -c | sort -nr | awk '{print $2" "$1}' >$OUT_BIN/dict.$1.txt
  #  cat $STATF | tr ' ' '\n' | sort | uniq -c | sort -nr | awk '{print $2" "$1}' >$OUT_BIN/dict.$1.txt
}

echo
binarize_parallel $L1 $L2
echo
binarize_parallel $L2 $L1

echo
binarize_monolingual $L1
echo
binarize_monolingual $L2

cp $SPM_PREFIX.dict.txt $OUT_BIN/dict.txt
cp $SPM_PREFIX.model $OUT_BIN/spm.model
cp $SPM_PREFIX.vocab $OUT_BIN/spm.vocab
