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
CONDA_ENV="nmt-pretrain" # This is the name of the project's conda environment
ACCOUNT="Project123-GPU" # Your slurm account.
TIME="35:59:59"          # The duration of each slurm job. E.g.
ARRAY="1-4%1"            # How many times to repeat the slurm job."1-2%1"

############################################################################
# Job Generator
############################################################################
generate_job() {
  PARA_DATA=$(readlink -f $ROOT/data/${1})
  SRC_LANG=${2}
  TRG_LANG=${3}
  GPUS=${4}
  SEED=${5}
  OPTIM=${6}
  PRETRAINED_HP=${7}
  CHECKPOINT=${8}

  EXP_NAME="supervised_nmt_${SRC_LANG}${TRG_LANG}"
  if [ ! -z "$CHECKPOINT" ]; then
    EXP_NAME+=".transfer_from.$(basename "$(dirname "$CHECKPOINT")")"
  fi
  EXP_NAME+="_seed=${SEED}"

  SAVE_DIR=$ROOT/checkpoints/supervised_nmt/$EXP_NAME
  mkdir -p "$ROOT/checkpoints/supervised_nmt/$EXP_NAME"
  LAUNCH_TRAIN="$SAVE_DIR/train.sh"
  LAUNCH_EVAL="$SAVE_DIR/eval.sh"

  echo "--------------------------------------------------------------------------"
  echo "Generating launcher for experiment '$EXP_NAME'"
  echo "SAVE_DIR:$SAVE_DIR"
  echo "TRAIN-LAUNCHER:$LAUNCH_TRAIN"
  echo "EVAL-LAUNCHER:$LAUNCH_EVAL"
  echo "--------------------------------------------------------------------------"
  echo

  #--------------------------------
  # Create train launcher
  #--------------------------------
  cat <<END >$LAUNCH_TRAIN
#!/bin/bash
#SBATCH -A $ACCOUNT
#SBATCH --job-name=${EXP_NAME}
#SBATCH --output=$SAVE_DIR/train.%j.out
#SBATCH --error=$SAVE_DIR/train.%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:$GPUS
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=$TIME
#SBATCH --partition=pascal
#SBATCH --array=$ARRAY

# prepare for experiment - load necessary modules etc.
source $HOME/.bashrc
conda activate $CONDA_ENV

RESTORE=""

END

  if [ ! -z "$CHECKPOINT" ]; then

    cat <<END >>$LAUNCH_TRAIN

if [ ! -f "$SAVE_DIR/checkpoint_best.pt" ]; then
  RESTORE="--restore-file $CHECKPOINT --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler"
fi

END

  fi

  cat <<END >>$LAUNCH_TRAIN
fairseq-train $PARA_DATA \\
  --user-dir $ROOT/user \\
  --source-lang $SRC_LANG --target-lang $TRG_LANG \\
  --arch marss_analysis \\
  --dropout 0.3 --attention-dropout 0.1 \\
  --criterion label_smoothed_cross_entropy \\
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-06 \\
  --clip-norm 0.0 \\
  --ddp-backend no_c10d \\
  --max-source-positions 256 \\
  --max-target-positions 256 \\
  --save-dir $SAVE_DIR \\
  --tensorboard-logdir $TB_DIR/$EXP_NAME \\
  --log-interval 100  --log-format tqdm \\
  --save-interval-updates 5000 \\
  --keep-best-checkpoints 1 \\
  --keep-interval-updates 1 \\
  --save-interval 1000 \\
  --validate-interval 1000 \\
  --no-epoch-checkpoints \\
  --eval-bleu \\
  --eval-bleu-args '{"beam": 5}' \\
  --eval-bleu-detok space \\
  --eval-bleu-remove-bpe sentencepiece \\
  --eval-bleu-print-samples \\
  --best-checkpoint-metric bleu \\
  --maximize-best-checkpoint-metric \\
  --seed $SEED \\
  $OPTIM $PRETRAINED_HP \\
  \$RESTORE

sbatch $LAUNCH_EVAL

scancel \${SLURM_ARRAY_JOB_ID}

END

  #--------------------------------
  # Create evaluation launcher
  #--------------------------------

  if [ "${SRC_LANG}" == "de" ] || [ "${TRG_LANG}" == "de" ]; then
    TESTSETS="$ROOT/data/deen/parallel/newstest2017-${SRC_LANG}${TRG_LANG}"
    TESTSETS+=" $ROOT/data/deen/parallel/newstest2018-${SRC_LANG}${TRG_LANG}"
    TESTSETS+=" $ROOT/data/deen/parallel/newstest2019-${SRC_LANG}${TRG_LANG}"
  elif [ "${SRC_LANG}" == "ne" ] || [ "${TRG_LANG}" == "ne" ]; then
    TESTSETS="$ROOT/data/flores/wikipedia_en_ne_si_test_sets/wikipedia.dev.ne-en"
    TESTSETS="$ROOT/data/flores/wikipedia_en_ne_si_test_sets/wikipedia.devtest.ne-en"
    TESTSETS="$ROOT/data/flores/wikipedia_en_ne_si_test_sets/wikipedia.test.ne-en"
  elif [ "${SRC_LANG}" == "si" ] || [ "${TRG_LANG}" == "si" ]; then
    TESTSETS="$ROOT/data/flores/wikipedia_en_ne_si_test_sets/wikipedia.dev.si-en"
    TESTSETS="$ROOT/data/flores/wikipedia_en_ne_si_test_sets/wikipedia.devtest.si-en"
    TESTSETS="$ROOT/data/flores/wikipedia_en_ne_si_test_sets/wikipedia.test.si-en"
  fi

  if [[ TRG_LANG == "ne" || TRG_LANG == "si" ]]; then
    EVAL_TOKENIZER="-tok none -s none"
  else
    EVAL_TOKENIZER="-l ${SRC_LANG}-${TRG_LANG}"
  fi

  cat <<END >$LAUNCH_EVAL
#!/bin/bash
#SBATCH -A $ACCOUNT
#SBATCH --job-name=${EXP_NAME}
#SBATCH --output=$SAVE_DIR/eval.%j.out
#SBATCH --error=$SAVE_DIR/eval.%j.err
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:01:00
#SBATCH --partition=pascal

# prepare for experiment - load necessary modules etc.
source $HOME/.bashrc
conda activate $CONDA_ENV

for testset in $TESTSETS; do

  test_prefix=\$(basename \$testset)
  OUTDIR=$SAVE_DIR/gen_outputs/\$test_prefix
  mkdir -p \$OUTDIR

  INPUT=\$testset.${SRC_LANG}
  REFERENCE=\$testset.${TRG_LANG}
  SPM_MODEL=$PARA_DATA/spm.model
  SPM_SRC=$ROOT/tools/sentencepiece/build/src

  cat \$INPUT | "\$SPM_SRC/spm_encode" --model=\$SPM_MODEL --output_format=piece | fairseq-interactive $PARA_DATA \\
    --user-dir $ROOT/user \\
    --source-lang $SRC_LANG --target-lang $TRG_LANG \\
    --path $SAVE_DIR/checkpoint_best.pt \\
    $PRETRAINED_HP \\
    --results-path $SAVE_DIR \\
    --beam 5 \\
    --remove-bpe 'sentencepiece' \\
    --sacrebleu \\
    --buffer-size 64 \\
    --max-tokens 20000 \\
    --results-path \${OUTDIR} | tee \${OUTDIR}/generate.txt

  cat \${OUTDIR}/generate.txt | grep -P "^D-" | sort -nr -k1.2 | cut -f3 >\${OUTDIR}/gen_best.output
  cat \${OUTDIR}/gen_best.output | sacrebleu \$REFERENCE $EVAL_TOKENIZER >\${OUTDIR}/bleu.results
  cat \${OUTDIR}/gen_best.output | sacrebleu \$REFERENCE $EVAL_TOKENIZER -b -m bleu -w 5 >\${OUTDIR}/gen_best.bleu
  cat \${OUTDIR}/gen_best.output | sacrebleu \$REFERENCE $EVAL_TOKENIZER -b -m chrf -w 5 >\${OUTDIR}/gen_best.chrf

done

END

  sbatch $LAUNCH_TRAIN
}

# ---------------------------------------------------------------------
# Random Initialization Experiments
# ---------------------------------------------------------------------
RANDOM_OPTIM="--lr 5e-4"
RANDOM_OPTIM+=" --lr-scheduler inverse_sqrt"
RANDOM_OPTIM+=" --warmup-updates 8000"
RANDOM_OPTIM+=" --max-update 50000"
RANDOM_OPTIM+=" --max-tokens 6000"
RANDOM_OPTIM+=" --update-freq 2"
RANDOM_OPTIM+=" --label-smoothing 0.1"

declare -a languages
languages+=('deen/parallel_bin;de;en')
languages+=('flores/parallel_neen_bin;ne;en')
languages+=('flores/parallel_sien_spm;si;en')

for comb in "${languages[@]}"; do
  IFS=";" read -r -a arr <<<"${comb}"
  DATASET="${arr[0]}"
  SRC="${arr[1]}"
  TGT="${arr[2]}"

  for seed in 1 2 3; do

    # generate_job PARA_DATA, SRC_LANG, TRG_LANG

    # XX->EN
    generate_job "${DATASET}" "${SRC}" "${TGT}" 2 $seed "$RANDOM_OPTIM" "" ""
    # EN->XX
    generate_job "${DATASET}" "${TGT}" "${SRC}" 2 $seed "$RANDOM_OPTIM" "" ""
  done

done

# ---------------------------------------------------------------------
# Finetuning Experiments
# ---------------------------------------------------------------------

PRETRAINED_OPTIM="--lr 3e-05"
PRETRAINED_OPTIM+=" --lr-scheduler fixed"
PRETRAINED_OPTIM+=" --warmup-updates 4000"
PRETRAINED_OPTIM+=" --max-update 80000"
PRETRAINED_OPTIM+=" --max-tokens 6000"
PRETRAINED_OPTIM+=" --update-freq 2"
PRETRAINED_OPTIM+=" --label-smoothing 0.1"

declare -a flores
flores+=('flores/parallel_neen_bin;ne;en')
flores+=('flores/parallel_sien_spm;si;en')

for comb in "${flores[@]}"; do
  IFS=";" read -r -a arr <<<"${comb}"
  DATASET="${arr[0]}"
  SRC="${arr[1]}"
  TGT="${arr[2]}"

  # IMPORTANT: update the model checkpoints with your own
  declare -a flores_models=()
  flores_models+=("$ROOT/checkpoints/pretraining/mbart.flores_${SRC}${TGT}/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_mask=35/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_replace=35/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_replace=35_ertd=6/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_shuffle=5/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_tied_replace=35/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_tied_replace=35_ertd=6/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_tied_replace=35_maskdecoder/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_tied_replace=35_multitask/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_tied_replace=35_replacedecoder/checkpoint_best.pt")
  flores_models+=("$ROOT/checkpoints/pretraining/marss.flores_${SRC}${TGT}_tied_replace=35_shuffle=3/checkpoint_best.pt")

  for seed in 1 2 3; do
    for cp in "${flores_models[@]}"; do

      PRETRAINED_PARAMS="--task translation_from_pretrained_bart --langs ${SRC},${TGT} --prepend-bos"

      # XX->EN
      generate_job "${DATASET}" "${SRC}" "${TGT}" 2 $seed "$PRETRAINED_OPTIM" "$PRETRAINED_PARAMS" "$cp"
      # EN->XX
      generate_job "${DATASET}" "${TGT}" "${SRC}" 2 $seed "$PRETRAINED_OPTIM" "$PRETRAINED_PARAMS" "$cp"

    done
  done

done

# IMPORTANT: update the model checkpoints with your own
declare -a deen_models=()
deen_models+=("$ROOT/checkpoints/pretraining/mbart.deen.analysis/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_mask=15/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_mask=35/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_mask=50/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_shuffle=3/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_shuffle=5/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=15/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=50/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35_ertd=4/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35_ertd=6/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_ratio=0.5_replace=35/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_ratio=1.0_replace=35/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_ertd=4/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_ertd=6/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_topp=0.9_replace=35/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_multitask/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_shuffle=3/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35_shuffle=3_ertd=4/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35_shuffle=3_ertd=6/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_maskdecoder/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_replacedecoder/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_shuffle=3_ertd=4/checkpoint_best.pt")
deen_models+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_shuffle=3_ertd=6/checkpoint_best.pt")

for seed in 1 2 3; do
  for cp in "${deen_models[@]}"; do

    PRETRAINED_PARAMS="--task translation_from_pretrained_bart --langs ${SRC},${TGT} --prepend-bos"

    # XX->EN
    generate_job "deen/parallel_bin" "de" "en" 2 $seed "$PRETRAINED_OPTIM" "$PRETRAINED_PARAMS" "$cp"
    # EN->XX
    generate_job "deen/parallel_bin" "en" "de" 2 $seed "$PRETRAINED_OPTIM" "$PRETRAINED_PARAMS" "$cp"

  done
done
