#!/bin/bash

set -e

# data path
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CURRENT_PATH=$(readlink -f "$DIR")
TOOLS_PATH=$CURRENT_PATH/tools

# tools path
mkdir -p $TOOLS_PATH

# tools
MOSES_DIR=$TOOLS_PATH/mosesdecoder

#
# Download and install tools
#

cd $TOOLS_PATH

# Download Moses
if [ ! -d "$MOSES_DIR" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

# Download Sentencepiece
if [ ! -d "$TOOLS_PATH/sentencepiece" ]; then
  echo "Cloning sentencepiece from GitHub repository..."
  git clone https://github.com/google/sentencepiece.git
  cd sentencepiece
  mkdir build
  cd build
  cmake ..
  make -j $(nproc)
  cd $TOOLS_PATH
fi

# Download BTExtractor
if [ ! -d $TOOLS_PATH/extract_bt_data.py ]; then
  cd $TOOLS_PATH
  echo "Cloning WikiExtractor from GitHub repository..."
  wget -c https://raw.githubusercontent.com/facebookresearch/fairseq/b5a039c292facba9c73f59ff34621ec131d82341/examples/backtranslation/extract_bt_data.py
  cd $CURRENT_PATH
fi
