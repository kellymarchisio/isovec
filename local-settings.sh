#!/bin/bash

###############################################################################
### Environment setting.
###############################################################################
# start conda env.
source /home/kmarc/anaconda3/etc/profile.d/conda.sh
conda activate /home/kmarc/anaconda3/envs/isovec

# path to my working directory
export DIR=`pwd`
echo 'Creating Local Temp'
export TEMPDIR=$DIR/tmp
export TMP=$DIR/tmp
export MOSES_SCRIPTS=/home/kmarc/moses/scripts
export VECMAP=$DIR/third_party/vecmap_fork
export WORD2VEC=$DIR/third_party/word2vec/word2vec
export ISOSTUDY_SCRIPTS=`pwd`/third_party/iso_study_fork/scripts
export GRID=CLSP

echo 'TEMPDIR is: ' $TEMPDIR
echo You are running in this environment: $CONDA_PREFIX
echo 'You are running on machine: ' `hostname` ' on the ' $GRID ' grid.'
set -e
