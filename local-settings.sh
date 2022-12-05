#!/bin/bash

###############################################################################
### Environment setting.
###############################################################################

# path to my working directory
export DIR=`pwd`
echo 'Creating Local Temp'
export TEMPDIR=$DIR/tmp
export TMP=$DIR/tmp
export MOSES_SCRIPTS=$DIR/third_party/mosesdecoder/scripts
export VECMAP=$DIR/third_party/vecmap_fork
export WORD2VEC=$DIR/third_party/word2vec/word2vec
export ISOSTUDY_SCRIPTS=$DIR/third_party/iso_study_fork/scripts

echo 'TEMPDIR is: ' $TEMPDIR
set -e
