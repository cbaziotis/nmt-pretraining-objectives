#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
CURRENT_PATH=$(readlink -f "$DIR")

bash "$CURRENT_PATH/install-tools.sh"
MOSES=$CURRENT_PATH/tools/mosesdecoder
SGM2TEXT=$MOSES/scripts/ems/support/input-from-sgm.perl

MONO_PATH=$CURRENT_PATH/data/deen/mono
PARA_PATH=$CURRENT_PATH/data/deen/parallel
mkdir -p "$MONO_PATH"
mkdir -p "$PARA_PATH"

# seeding adopted from https://stackoverflow.com/a/41962458/7820599
get_seeded_random() {
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null
}

# Download Monolingual Data
for LANG in en de; do

  echo "Downloading $LANG monolingual data..."

  SUBSAMPLE=5000000

  if [ ! -f "$MONO_PATH/$LANG.txt" ]; then

    #for year in {2007..2017}; do
    for year in {2007..2008}; do
      FILENAME="news.${year}.${LANG}.shuffled.deduped.gz"
      wget -c "https://data.statmt.org/news-crawl/${LANG}/${FILENAME}" -P $MONO_PATH/
    done

    echo "Concatenating $LANG monolingual data..."
    cat $(ls $MONO_PATH/*.${LANG}.*) | gunzip >$MONO_PATH/$LANG.txt
    rm -rf $MONO_PATH/*.${LANG}.*gz

    echo "Sampling 5M sentences..."
    cat $MONO_PATH/$LANG.txt |
      awk 'NF > 7' |
      awk 'NF < 250' |
      shuf --random-source=<(get_seeded_random 452) |
      head -n $SUBSAMPLE >"$MONO_PATH/news_crawl.$LANG"

    rm "$MONO_PATH/$LANG.txt"

  else
    echo "Data already downloaded!"
  fi

done

# Download Parallel Data
wget -c http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz -P $PARA_PATH/
tar -xvzf $PARA_PATH/training-parallel-nc-v13.tgz -C $PARA_PATH/
mv $PARA_PATH/training-parallel-nc-v13/*de-en* /$PARA_PATH
rm -rf $PARA_PATH/training-parallel-nc-v13
rm $PARA_PATH/training-parallel-nc-v13.tgz

echo "Downloading newstests..."
for year in 17 18 19; do
  wget -c http://data.statmt.org/wmt${year}/translation-task/test.tgz -O $PARA_PATH/wmt-test-${year}.tgz
  mkdir -p $PARA_PATH/wmt-test-${year}
  tar -xzf $PARA_PATH/wmt-test-${year}.tgz --directory $PARA_PATH/wmt-test-${year}
  mv $PARA_PATH/wmt-test-${year}/*/*deen* $PARA_PATH/
  mv $PARA_PATH/wmt-test-${year}/*/*ende* $PARA_PATH/
  rm $PARA_PATH/wmt-test-${year}.tgz
  rm -rf $PARA_PATH/wmt-test-${year}

  L1=de
  L2=en
  eval "$SGM2TEXT" <"$PARA_PATH/newstest20${year}-${L1}${L2}-src.${L1}.sgm" >"$PARA_PATH/newstest20${year}-${L1}${L2}.${L1}"
  eval "$SGM2TEXT" <"$PARA_PATH/newstest20${year}-${L1}${L2}-ref.${L2}.sgm" >"$PARA_PATH/newstest20${year}-${L1}${L2}.${L2}"
  eval "$SGM2TEXT" <"$PARA_PATH/newstest20${year}-${L2}${L1}-src.${L2}.sgm" >"$PARA_PATH/newstest20${year}-${L2}${L1}.${L2}"
  eval "$SGM2TEXT" <"$PARA_PATH/newstest20${year}-${L2}${L1}-ref.${L1}.sgm" >"$PARA_PATH/newstest20${year}-${L2}${L1}.${L1}"

  rm $PARA_PATH/*.sgm
done
