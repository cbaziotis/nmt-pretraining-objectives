# Description

This repository contains the source code and data for the paper: "Exploration of Unsupervised Pretraining Objectives for
Machine Translation" in Findings of ACL 2021.

(code and data will be uploaded soon)

# 1. Setup Development Environment

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

# 2. Data

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


# 3. Launching Experiments


The `experiments/` directory contains scripts that launch experiments, such as pretraining or supervised NMT.
Each experiment launcher generates the scripts for training/evaluating each particular model.  
```text
experiments/
├── launch_pretraining.sh       # Launches the jobs for the pretraining models
├── launch_supervised_nmt.sh    # Launches the jobs for training supervised NMT models
├── launch_unsupervised_nmt.sh  # Launches the jobs for training unsupervised NMT models
└── analysis/                   # Contains launchers for experiments/probes for analyzing pretrained models
```

##### Example: How to train the pretrained models
This is the process for training the pretraining models.
You can follow the same process for the running other experiments, such (un)supervised NMT.
First, execute the experiment launcher:
```shell
bash experiments/launch_pretraining.sh
```
This script will generate all the (SLURM) runner scripts for each pretraining models.
If you execute the script in a cluster that uses SLURM then it will also submit a job for each model.

Once you execute the script, it will create the directories for each model under 
`checkpoints/pretraining/MODEL`,
and will also save the runner script that will be used for training each model 
(e.g. `checkpoints/pretraining/marss.deen.analysis_mask=15/train.sh`)

```shell
checkpoints/pretraining/
├── marss.deen.analysis_mask=15
│   └── train.sh
├── marss.deen.analysis_mask=35
│   └── train.sh
├── marss.deen.analysis_mask=50
│   └── train.sh
...
```
This folder will be used during for storing all the training and validation data for each model,
such as model checkpoints, training logs, model outputs etc.


**About the runner scripts.** Each `train.sh` contains the command for training each model.
Also, each can be submitted to SLURM by running `sbatch train.sh`. Please inspect the files and see their SLURM headers.
If you do use SLURM, please update the SLURM header parameters in each experiment launcher that generates the model runner scripts 
based on your server settings.
If you don't use SLURM you can directly run it as any other script on your machine, i.e., `bash train.sh`.

```shell
CONDA_ENV="nmt-pretrain"  # This is the name of the project's conda environment
ACCOUNT="Project123-GPU"      # Your slurm account.
TIME="35:59:59"               # The duration of each slurm job. E.g.
ARRAY="1-4%1"                 # How many times to repeat the slurm job."1-2%1"
```

### Start Tensorboard server (optional)

The training progress (statistics, losses etc.) of each model will be saved
under `runs/`. To visualize the training process run:

```shell script
tensorboard --logdir runs/ --port 8002
``` 

and open `http://localhost:8002/` in your browser.


# Pretrained Models

You can download the pretrained models and inspect their analysis data
in https://data.statmt.org/cbaziotis/projects/acl2021-nmt-pretraining/checkpoints/.


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
