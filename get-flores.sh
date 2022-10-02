#!/bin/bash

# -------------------------------------------------------------------------
# The code to download the flores v1 data is currently broken
#
# git clone https://github.com/facebookresearch/flores.git data/flores
# bash data/flores/floresv1/download-data.sh # this no longer works!
#
# This script downloads the FLORESv1 data that were used for the paper
# from our university servers.
# -------------------------------------------------------------------------

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CURRENT_PATH=$(readlink -f "$DIR")

DATA_PATH=$CURRENT_PATH/data/flores
mkdir -p "$DATA_PATH"

# in statmt
# tar -czf floresv1_data.tar.gz all-clean-hi all-clean-ne all-clean-si wikipedia_en_ne_si_test_sets mono/mono.norm.sample5000000.ne mono/mono.norm.sample5000000.si mono/mono.sample5000000.en wiki_ne_en_bpe5000 wiki_si_en_bpe5000
# mv floresv1_data.tar.gz ~/statmt/projects/acl2021-nmt-pretraining/

wget -c https://data.statmt.org/cbaziotis/projects/acl2021-nmt-pretraining/floresv1_data.tar.gz -P $DATA_PATH
tar -xzf $DATA_PATH/floresv1_data.tar.gz --directory $DATA_PATH
rm $DATA_PATH/floresv1_data.tar.gz