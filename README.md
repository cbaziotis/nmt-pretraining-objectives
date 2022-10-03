# Description

This repository contains the source code and data for the paper: "Exploration of Unsupervised Pretraining Objectives for
Machine Translation" in Findings of ACL 2021.

(code and data will be uploaded soon)

### Configuration of Development Environment

1. __Create Conda environment.__
   Use this environment for the project.

```shell
conda create -n nmt-pretrain python=3.7
conda activate nmt-pretrain

```

2. __Install PyTorch.__
   The project was developed with pytorch v1.6.0.
   For different cuda versions see https://pytorch.org/get-started/previous-versions/#v140.

```shell
 pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

3. __Install custom fairseq.__
   While implementing the models of the paper, fairseq had some bugs that prevented training mBART.
   Run the following commands to install fairseq v9.0.0 with our fixes (see
   branch https://github.com/cbaziotis/fairseq/tree/nmt_pretrain_acl21)

```shell
LIB_PATH=../fairseq_paper_acl21 # preferanbly choose a path outside from the current project
#git clone --single-branch --branch nmt_pretrain_acl21 https://github.com/cbaziotis/fairseq.git $LIB_PATH
cd $LIB_PATH
python -m pip install --editable .
python setup.py build_ext --inplace
```

4. __Install remaining requirements.__

```shell
pip install -r requirements.txt
```

### Preparing the Data

1. __Download the data.__
   For FLORESv1, we provide the data from our servers because the download scripts
   (https://github.com/facebookresearch/flores/blob/main/previous_releases/floresv1/download-data.sh)
   from the flores repository are no longer working.

```shell
bash get-ende.sh
bash get-flores.sh
```

2. __(OPTIONAL) Preprocess data for pretraining.__
   If you want to re-use the pretrained models,
   you will need to preprocess the data with the Sentencepiece (SPM) models
   that were use in the paper.
   Execute the following command to download the pretrained SPM models.
   The preprocessing code will use these if they are available,
   otherwise new SPM models will be trained.

```shell
DATA_PATH=./data
wget -c https://data.statmt.org/cbaziotis/projects/acl2021-nmt-pretraining/pretrain_spms.tar.gz -P $DATA_PATH/spm
tar -xzf $DATA_PATH/spm/pretrain_spms.tar.gz --directory $DATA_PATH/spm
rm $DATA_PATH/spm/pretrain_spms.tar.gz

```

3. __Preprocess data for pretraining.__
   This step tokenizes and binarizes the monolingual data  
   for each language group for unsupervised pretraining.

```shell
bash prepare_pretraining.sh deen
bash prepare_pretraining.sh neen
bash prepare_pretraining.sh sien
```

4. __Preprocess data for supervised NMT.__
   This step tokenizes and binarizes the parallel data  
   of each language pair for supervised NMT training.
   This step reuses the SPM model that was used in step1.

```shell
bash prepare_supervised_nmt.sh deen
bash prepare_supervised_nmt.sh neen
bash prepare_supervised_nmt.sh sien
```

5. __Preprocess data for unsupervised NMT.__
   This step re-uses and prepares the monolingual data from step2
   for the unsupervised NMT experiments.

```shell
bash prepare_unsupervised_nmt.sh deen
bash prepare_unsupervised_nmt.sh neen
bash prepare_unsupervised_nmt.sh sien
```

# Paper Citation

```
@proceedings{baziotis-2021-pretraining-obectives,
    title = "Exploration of Unsupervised Pretraining Objectives for Machine Translation",
    editor = "Baziotis, Christos  and
      Titov, Ivan  and
      Birch, Alexandra  and
      Haddow, Barry",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "",
}
```
