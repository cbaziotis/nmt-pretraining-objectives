#!/bin/bash

############################################################################
# CONFIG
############################################################################
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT=$(readlink -f "$DIR/..")
TB_DIR=$(readlink -f "$HOME/workspace/tensorboard")

############################################################################
# SLURM SETTINGS - Update these parameters based on your setup/server
############################################################################
CONDA_ENV="nmt-pretrain"  # This is the name of the project's conda environment
ACCOUNT="Project123-GPU"      # Your slurm account.
TIME="35:59:59"               # The duration of each slurm job. E.g.
ARRAY="1-4%1"                 # How many times to repeat the slurm job."1-2%1"

############################################################################
# Job Generator
############################################################################
generate_job() {
  #  1=MODEL, 2=SRC_LANG, 3=TRG_LANG, 4=PARA_DATA, 5=BT_DATA
  EXP_NAME=${1}
  LANGS=${2}
  DATA=$(readlink -f $ROOT/data/${3})
  TASK=${4}
  ARCH=${5}
  CRITERION=${6}
  MASKING=${7}

  TOTAL_UPDATES=300000 # Total number of training steps
  WARMUP_UPDATES=16000 # Warmup the learning rate over this many updates
  MAX_TOKENS=4000
  PEAK_LR=0.0005       # Peak learning rate, adjust as needed
  WEIGHT_DECAY=0.0
  CLIP_NORM=0.0
  EPS=1e-06
  BETA='(0.9, 0.999)'
  UPDATE_FREQ=2 # Increase the batch size X

  SAVE_DIR=$ROOT/checkpoints/pretraining/$EXP_NAME
  mkdir -p "$ROOT/checkpoints/pretraining/$EXP_NAME"
  FILE="$SAVE_DIR/train.sh"

  echo "--------------------------------------------------------------------------"
  echo "Generating launcher for experiment '$EXP_NAME'"
  echo "NOISE: '$MASKING'"
  echo "SAVE_DIR:$SAVE_DIR"
  echo "LAUNCHER:$FILE"
  echo "--------------------------------------------------------------------------"
  echo

  cat <<END >$FILE
#!/bin/bash
#SBATCH -A $ACCOUNT
#SBATCH --job-name=${EXP_NAME}
#SBATCH --output=$SAVE_DIR/train.%j.out
#SBATCH --error=$SAVE_DIR/train.%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=$TIME
#SBATCH --partition=pascal
#SBATCH --array=$ARRAY

# prepare for experiment - load necessary modules etc.
source $HOME/.bashrc
conda activate $CONDA_ENV

fairseq-train $DATA \\
  --user-dir $ROOT/user \\
  --langs $LANGS --add-lang-token \\
  --bpe sentencepiece --sentencepiece-vocab $DATA/spm.model \\
  --task $TASK \\
  --arch $ARCH \\
  --criterion $CRITERION \\
  $MASKING \\
  --optimizer adam --adam-betas '$BETA' --adam-eps $EPS \\
  --warmup-updates $WARMUP_UPDATES \\
  --max-update $TOTAL_UPDATES \\
  --lr $PEAK_LR \\
  --lr-scheduler inverse_sqrt \\
  --max-tokens $MAX_TOKENS --update-freq $UPDATE_FREQ  \\
  --weight-decay $WEIGHT_DECAY --clip-norm $CLIP_NORM \\
  --max-source-positions 256 --max-target-positions 256 \\
  --sample-break-mode eos \\
  --save-dir $SAVE_DIR \\
  --tensorboard-logdir $TB_DIR/$EXP_NAME \\
  --log-interval 100 --log-format tqdm \\
  --no-epoch-checkpoints \\
  --save-interval-updates 10000 \\
  --keep-interval-updates 0 \\
  --patience 10 \\
  --skip-invalid-size-inputs-valid-test \\
  --ddp-backend no_c10d --num-workers 0

scancel \$SLURM_ARRAY_JOB_ID

END

  sbatch $FILE
}

REPLACE_WORD='--mask 0.35 --replace-length -1 --rotate 0 --mask-replace --mask-length word'
REPLACE_SPAN='--mask 0.35 --replace-length -1 --rotate 0 --mask-replace --mask-length span-poisson'
# generate_job 1=EXP_NAME, 2=LANGS, 3=DATA, 4=TASK, 5=ARCH, 6=CRITERION, 7=NOISING


# ENDE
DATASET=deen/mono_bin
TAG=deen.analysis
LANG_PAIR='de,en'

#-------------------------------------------------------------------------------------------------
#EXP-1: test the impact of having a stronger generator
#-------------------------------------------------------------------------------------------------
# 1) Larger generator
# 2) Constraint sampling to avoid sampling from the tail of the generator
#-------------------------------------------------------------------------------------------------
generate_job marss."${TAG}"_ratio=0.5_replace=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 0.5' "$REPLACE_WORD"
generate_job marss."${TAG}"_ratio=1.0_replace=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 1.0' "$REPLACE_WORD"
generate_job marss."${TAG}"_tied_replace=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis 'marss --generator-ratio 1 --tie-generator-encoder' \
  '--mask 0.35 --mask-length word --replace-length -1 --rotate 0 --mask-replace'
generate_job marss."${TAG}"_tied_topp=0.9_replace=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 1.0 --tie-generator-encoder' "$REPLACE_WORD --mask-replace-sampling-topp 0.9"


#-------------------------------------------------------------------------------------------------
# EXP-2: test mBART + with MLM loss
# the replacements are not actually added, we just consider the MLM loss and use the original input
#-------------------------------------------------------------------------------------------------
generate_job mbart."${TAG}" $LANG_PAIR $DATASET multilingual_denoising marss_analysis cross_entropy \
  "--mask 0.35 --mask-length span-poisson --replace-length -1 --rotate 0"

generate_job marss."${TAG}"_tied_replace=35_multitask $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --test-multitask"

#-------------------------------------------------------------------------------------------------
#EXP-3: test the impact of having a discriminator over the middle layers of the encoder
#-------------------------------------------------------------------------------------------------
# 1) Test if using the middle layer is better than the last
# 2) Test how the RTD works with shuffling
#-------------------------------------------------------------------------------------------------
generate_job marss."${TAG}"_replace=35_ertd=6 $LANG_PAIR $DATASET marss_pretraining \
  "marss_analysis --replacement-detection-encoder 6 --rtd-loss-coefficient 25" \
  'marss --generator-ratio 1' "$REPLACE_WORD"
generate_job marss."${TAG}"_replace=35_ertd=4 $LANG_PAIR $DATASET marss_pretraining \
  "marss_analysis --replacement-detection-encoder 4 --rtd-loss-coefficient 25" \
  'marss --generator-ratio 1' "$REPLACE_WORD"
generate_job marss."${TAG}"_replace=35_shuffle=3_ertd=6 $LANG_PAIR $DATASET marss_pretraining \
  "marss_analysis --replacement-detection-encoder 6 --rtd-loss-coefficient 25" \
  'marss --generator-ratio 1' "$REPLACE_WORD --word-shuffle 3"
generate_job marss."${TAG}"_replace=35_shuffle=3_ertd=4 $LANG_PAIR $DATASET marss_pretraining \
  "marss_analysis --replacement-detection-encoder 4 --rtd-loss-coefficient 25" \
  'marss --generator-ratio 1' "$REPLACE_WORD --word-shuffle 3"
generate_job marss."${TAG}"_tied_replace=35_ertd=6 $LANG_PAIR $DATASET marss_pretraining \
  "marss_analysis --replacement-detection-encoder 6 --rtd-loss-coefficient 25" \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD"
generate_job marss."${TAG}"_tied_replace=35_ertd=4 $LANG_PAIR $DATASET marss_pretraining \
  "marss_analysis --replacement-detection-encoder 4 --rtd-loss-coefficient 25" \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD"
generate_job marss."${TAG}"_tied_replace=35_shuffle=3_ertd=6 $LANG_PAIR $DATASET marss_pretraining \
  "marss_analysis --replacement-detection-encoder 6 --rtd-loss-coefficient 25" \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --word-shuffle 3"
generate_job marss."${TAG}"_tied_replace=35_shuffle=3_ertd=4 $LANG_PAIR $DATASET marss_pretraining \
  "marss_analysis --replacement-detection-encoder 4 --rtd-loss-coefficient 25" \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --word-shuffle 3"

#-------------------------------------------------------------------------------------------------
# EXP-4: test noise in the decoder noise
#-------------------------------------------------------------------------------------------------
# 1) Add only masking to the decoder to put pressure to encoder
# 2) Add masked replacements to decoder to test if it mitigates exposure bias
# 3) Add an RTD head over the middle-last layers of the decoder
#-------------------------------------------------------------------------------------------------
generate_job marss."${TAG}"_tied_replace=35_maskdecoder $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --mask-decoder 0.15"
generate_job marss."${TAG}"_tied_replace=35_replacedecoder $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --mask-decoder 0.15 --mask-replace-decoder"
generate_job marss."${TAG}"_tied_replace=35_replacedecoder_drtd=4 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --mask-decoder 0.15 --mask-replace-decoder --replacement-detection-decoder 4  --rtd-loss-coefficient 25"
generate_job marss."${TAG}"_tied_replace=35_replacedecoder_drtd=6 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --mask-decoder 0.15 --mask-replace-decoder --replacement-detection-decoder 6  --rtd-loss-coefficient 25"

#-------------------------------------------------------------------------------------------------
# EXP-5: test the impact of just masking, replacing, shuffling
#-------------------------------------------------------------------------------------------------
# 1) Compare mask 15% vs 35% vs 50%
# 2) Compare mask-replace 15% vs 35% vs 50%
# 3) Compare ONLY shuffling k=3 va k=5
#-------------------------------------------------------------------------------------------------

generate_job marss."${TAG}"_mask=15 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 0.5' "--mask 0.15 --mask-length word --replace-length -1 --rotate 0"
generate_job marss."${TAG}"_mask=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 0.5' "--mask 0.35 --mask-length word --replace-length -1 --rotate 0"
generate_job marss."${TAG}"_mask=50 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 0.5' "--mask 0.50 --mask-length word --replace-length -1 --rotate 0"

generate_job marss."${TAG}"_replace=15 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 0.5' "--mask 0.15 --mask-length word --replace-length -1 --rotate 0 --mask-replace"
generate_job marss."${TAG}"_replace=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 0.5' "--mask 0.35 --mask-length word --replace-length -1 --rotate 0 --mask-replace"
generate_job marss."${TAG}"_replace=50 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 0.5' "--mask 0.50 --mask-length word --replace-length -1 --rotate 0 --mask-replace"

generate_job marss."${TAG}"_shuffle=3 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  marss "--mask 0.0 --word-shuffle 3 --mask-length word --replace-length -1"
generate_job marss."${TAG}"_shuffle=5 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  marss "--mask 0.0 --word-shuffle 5 --mask-length word --replace-length -1"

generate_job marss."${TAG}"_tied_replace=35_shuffle=3 $LANG_PAIR $DATASET marss_pretraining marss_analysis \
  'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --word-shuffle 3"



declare -a languages
languages[0]='flores/mono_neen_bin;ne,en;flores_neen'
languages[1]='flores/mono_neen_bin;si,en;flores_sien'

for comb in "${languages[@]}"; do
  # turn e.g. 'a;b;c' into  array ['a', 'b', 'c']
  IFS=";" read -r -a arr <<<"${comb}"
  DATASET="${arr[0]}"
  LANG_PAIR="${arr[1]}"
  TAG="${arr[2]}"
  echo "---------------------------------------------------------"
  echo "DATASET : ${DATASET}"
  echo "LANG_PAIR : ${LANG_PAIR}"
  echo "TAG : ${TAG}"
  echo

  #1=EXP_NAME, 2=LANGS, 3=DATA, 4=TASK, 5=ARCH, 6=CRITERION, 7=MASKING
  #-------------------------------------------------------------------------------------------------
  # test mBART + with MLM loss
  # the replacements are not actually added, we just consider the MLM loss and use the original input
  #-------------------------------------------------------------------------------------------------
  generate_job mbart."${TAG}" $LANG_PAIR $DATASET multilingual_denoising marss_analysis cross_entropy "--mask 0.35 --mask-length span-poisson --replace-length -1 --rotate 0"
  generate_job marss."${TAG}"_tied_replace=35_multitask $LANG_PAIR $DATASET marss_pretraining marss_analysis 'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --test-multitask"

  #-------------------------------------------------------------------------------------------------
  # test noise in the decoder noise
  #-------------------------------------------------------------------------------------------------
  generate_job marss."${TAG}"_tied_replace=35_maskdecoder $LANG_PAIR $DATASET marss_pretraining marss_analysis 'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --mask-decoder 0.25"
  generate_job marss."${TAG}"_tied_replace=35_replacedecoder $LANG_PAIR $DATASET marss_pretraining marss_analysis 'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --mask-decoder 0.25 --mask-replace-decoder"

  #-------------------------------------------------------------------------------------------------
  # test the impact of just masking, replacing, shuffling
  #-------------------------------------------------------------------------------------------------
  generate_job marss."${TAG}"_mask=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis 'marss --generator-ratio 0.5' "--mask 0.35 --mask-length word --replace-length -1 --rotate 0"
  generate_job marss."${TAG}"_replace=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis       'marss --generator-ratio 0.5'                         "--mask 0.35 --mask-length word --replace-length -1 --rotate 0 --mask-replace"
  generate_job marss."${TAG}"_tied_replace=35 $LANG_PAIR $DATASET marss_pretraining marss_analysis  'marss --generator-ratio 1.0 --tie-generator-encoder' "--mask 0.35 --mask-length word --replace-length -1 --rotate 0 --mask-replace"
  generate_job marss."${TAG}"_shuffle=5 $LANG_PAIR $DATASET marss_pretraining marss_analysis marss "--mask 0.0 --word-shuffle 5 --mask-length word --replace-length -1"
  generate_job marss."${TAG}"_tied_replace=35_shuffle=3 $LANG_PAIR $DATASET marss_pretraining marss_analysis 'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD --word-shuffle 3"
  generate_job marss."${TAG}"_replace=35_ertd=6 $LANG_PAIR $DATASET marss_pretraining "marss_analysis --replacement-detection-encoder 6 --rtd-loss-coefficient 25" 'marss --generator-ratio 0.5' "$REPLACE_WORD"
  generate_job marss."${TAG}"_tied_replace=35_ertd=6 $LANG_PAIR $DATASET marss_pretraining "marss_analysis --replacement-detection-encoder 6 --rtd-loss-coefficient 25" 'marss --generator-ratio 1 --tie-generator-encoder' "$REPLACE_WORD"

done
