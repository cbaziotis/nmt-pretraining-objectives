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
  PARA_DATA=$(readlink -f $ROOT/data/deen/parallel_bin)
  MONO_DATA=$(readlink -f $ROOT/data/deen/mono_unmt_bin)
  SRC_LANG=en
  TRG_LANG=de
  GPUS=4
  CHECKPOINT=$1

  TOTAL_UPDATES=80000 # Total number of training steps
  WARMUP_UPDATES=2500 # Warmup the learning rate over this many updates
  PEAK_LR=3e-05       # Peak learning rate, adjust as needed
  UPDATE_FREQ=1       # Increase the batch size X
  MAX_TOKENS=5000
  MAX_TOKENS_VALID=1000

  EXP_NAME="unsupervised_nmt_${SRC_LANG}${TRG_LANG}"
  EXP_NAME+=".transfer_from.$(basename "$(dirname "$CHECKPOINT")")"

  SAVE_DIR=$ROOT/checkpoints/unsupervised/$EXP_NAME
  mkdir -p "$ROOT/checkpoints/unsupervised/$EXP_NAME"
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

if [ ! -f "$SAVE_DIR/checkpoint_best.pt" ]; then
  RESTORE="--restore-file $CHECKPOINT --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler"
else
  RESTORE=""
fi

fairseq-train $MONO_DATA \\
  --user-dir $ROOT/user \\
  --langs 'de,en' \\
  --arch marss_analysis \\
  --dropout 0.3 --attention-dropout 0.1 \\
  --task unsupervised_translation_from_pretrained \\
  \${RESTORE} \\
  --bt-beam-size 1 \\
  --bt-max-len-a 1.2 \\
  --bt-max-len-b 10 \\
  --bt-output-filtering-prob 0.01 \\
  --bt-output-filtering-steps 5000 \\
  --bt-no-repeat-ngram-size 0 \\
  --criterion cross_entropy \\
  --lr $PEAK_LR --lr-scheduler fixed --weight-decay 0.0 \\
  --optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-06 \\
  --warmup-updates $WARMUP_UPDATES \\
  --max-update $TOTAL_UPDATES \\
  --max-tokens $MAX_TOKENS \\
  --max-tokens-valid $MAX_TOKENS_VALID \\
  --update-freq $UPDATE_FREQ \\
  --ddp-backend no_c10d \\
  --max-source-positions 256 --max-target-positions 256 \\
  --save-dir $SAVE_DIR \\
  --tensorboard-logdir $TB_DIR/$EXP_NAME \\
  --log-interval 100 \\
  --validate-interval 1 \\
  --save-interval-updates 5000 \\
  --skip-invalid-size-inputs-valid-test \\
  --keep-interval-updates 0 \\
  --log-format tqdm \\
  --eval-bleu \\
  --eval-bleu-args '{"beam": 3, "max_len_a":1.3, "max_len_b":10}' \\
  --eval-bleu-detok space \\
  --eval-bleu-remove-bpe sentencepiece \\
  --eval-bleu-print-samples \\
  --best-checkpoint-metric bleu \\
  --no-epoch-checkpoints \\
  --maximize-best-checkpoint-metric \\
  --patience 5 \\
  --num-workers 0

sbatch $LAUNCH_EVAL

scancel \${SLURM_ARRAY_JOB_ID}

END

  #--------------------------------
  # Create evaluation launcher
  #--------------------------------


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


for i in "$SRC_LANG $TRG_LANG" "$TRG_LANG $SRC_LANG" ; do
  pair=(\$i)

  SRC=\${pair[0]}
  TGT=\${pair[1]}

  declare -a TESTSETS=()
  TESTSETS+=("$ROOT/data/deen/parallel/newstest2017-\${SRC}\${TGT}")
  TESTSETS+=("$ROOT/data/deen/parallel/newstest2018-\${SRC}\${TGT}")
  TESTSETS+=("$ROOT/data/deen/parallel/newstest2019-\${SRC}\${TGT}")

  for testset in "\${TESTSETS[@]}"; do

    test_prefix=\$(basename \$testset)
    OUTDIR=$SAVE_DIR/gen_outputs/\$test_prefix
    mkdir -p \$OUTDIR

    INPUT="\$testset.\${SRC}"
    REFERENCE="\$testset.\${TGT}"
    SPM_MODEL=$PARA_DATA/spm.model
    SPM_SRC=$ROOT/tools/sentencepiece/build/src

    cat \$INPUT | "\$SPM_SRC/spm_encode" --model=\$SPM_MODEL --output_format=piece | fairseq-interactive $PARA_DATA \\
      --user-dir $ROOT/user \\
      --task translation_from_pretrained_bart --langs de,en --prepend-bos \\
      --source-lang \$SRC --target-lang \$TGT \\
      --path $SAVE_DIR/checkpoint_best.pt \\
      --results-path $SAVE_DIR \\
      --beam 5 \\
      --remove-bpe 'sentencepiece' \\
      --sacrebleu \\
      --buffer-size 64 \\
      --max-tokens 20000 \\
      --results-path \${OUTDIR} | tee \${OUTDIR}/generate.txt

    cat \${OUTDIR}/generate.txt | grep -P "^D-" | sort -nr -k1.2 | cut -f3 >\${OUTDIR}/gen_best.output
    cat \${OUTDIR}/gen_best.output | sacrebleu \$REFERENCE -l "\${SRC}"-"\${TGT}" >\${OUTDIR}/bleu.results
    cat \${OUTDIR}/gen_best.output | sacrebleu \$REFERENCE -l "\${SRC}"-"\${TGT}" -b -m bleu -w 5 >\${OUTDIR}/gen_best.bleu
    cat \${OUTDIR}/gen_best.output | sacrebleu \$REFERENCE -l "\${SRC}"-"\${TGT}" -b -m chrf -w 5 >\${OUTDIR}/gen_best.chrf

  done

done


END

  sbatch $LAUNCH_TRAIN
}

# IMPORTANT: update the model checkpoints with your own
declare -a MODELS=()
MODELS+=("$ROOT/checkpoints/pretraining/mbart.deen.analysis/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_mask=15/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_mask=35/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_mask=50/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_shuffle=3/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_shuffle=5/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=15/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=50/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35_ertd=4/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35_ertd=6/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_ratio=0.5_replace=35/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_ratio=1.0_replace=35/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_ertd=4/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_ertd=6/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_topp=0.9_replace=35/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_multitask/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_shuffle=3/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35_shuffle=3_ertd=4/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_replace=35_shuffle=3_ertd=6/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_maskdecoder/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_replacedecoder/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_shuffle=3_ertd=4/checkpoint_best.pt")
MODELS+=("$ROOT/checkpoints/pretraining/marss.deen.analysis_tied_replace=35_shuffle=3_ertd=6/checkpoint_best.pt")

for model in "${MODELS[@]}"; do
  generate_job "$model"
done

