#!/bin/bash

############################################################################
# CONFIG
############################################################################
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT=$(readlink -f "$DIR/../..")

############################################################################
# SLURM SETTINGS - Update these parameters based on your setup/server
############################################################################
CONDA_ENV="nmt-pretrain" # This is the name of the project's conda environment
ACCOUNT="Project123-GPU" # Your slurm account.
TIME="00:01:00"          # The duration of each slurm job. E.g.

############################################################################
# Job Generator
############################################################################
generate_job() {
  SRC_LANG=${1}
  TRG_LANG=${2}
  LANGS='de,en'
  PARA_DATA=$(readlink -f $ROOT/data/deen/parallel_bin)
  SPM_MODEL=$PARA_DATA/spm.model
  MODEL=${3}
  INPUT_FILE=${4}
  TARGET_FILE=${5}
  DESC=${6}

  SAVE_DIR="$(dirname "$MODEL")/analysis"
  mkdir -p "$SAVE_DIR"
  EXP_NAME="sentence_similarity.$DESC.$(basename "$(dirname "$MODEL")")"
  LAUNCHER="$SAVE_DIR/sentence_similarity.$DESC.sh"

  cat <<END >$LAUNCHER.sh
  #!/bin/bash
  #SBATCH -A $ACCOUNT
  #SBATCH --job-name=${EXP_NAME}
  #SBATCH --output=$SAVE_DIR/sentence_similarity.%j.out
  #SBATCH --error=$SAVE_DIR/sentence_similarity.%j.err
  #SBATCH --ntasks=1
  #SBATCH --nodes=1
  #SBATCH --gres=gpu:1
  #SBATCH --ntasks-per-node=1
  #SBATCH --cpus-per-task=1
  #SBATCH --time=$TIME
  #SBATCH --partition=pascal

  # prepare for experiment - load necessary modules etc.
  source $HOME/.bashrc
  conda activate $CONDA_ENV

  python $ROOT/user/mask_replace_denoising/analysis/sentence_similarity.py $PARA_DATA \\
    --user-dir $ROOT/user \\
    --input $INPUT_FILE \\
    --target $TARGET_FILE \\
    --source-lang $SRC_LANG \\
    --target-lang $TRG_LANG \\
    --task translation_from_pretrained_bart --langs $LANGS --prepend-bos \\
    --bpe sentencepiece \\
    --sentencepiece-vocab $SPM_MODEL \\
    --path $MODEL \\
    --results-path $SAVE_DIR/sentence_similarity.$DESC.json \\
    --max-tokens 4000 \\
    --buffer-size 64

END

  sbatch $LAUNCHER.sh

}

# download the German-English Tatoeba test data
if [ ! -d "$ROOT/data/tatoeba" ]; then
  wget -c https://data.statmt.org/cbaziotis/projects/acl2021-nmt-pretraining/tatoeba_deu-eng_2020.tar.gz \
    -P $ROOT/data
  tar -xzf $ROOT/data/tatoeba_deu-eng_2020.tar.gz --directory $ROOT/data
  rm $ROOT/data/tatoeba_deu-eng_2020.tar.gz

fi

declare -a MODELS

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

  file_de=$(readlink -f $ROOT/data/tatoeba/tatoeba.deu-eng.deu.txt)
  file_en=$(readlink -f $ROOT/data/tatoeba/tatoeba.deu-eng.eng.txt)

  generate_job de en $model $file_de $file_en "tatoeba"

done
