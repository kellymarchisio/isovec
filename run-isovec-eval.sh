#!/bin/bash -v
. ./local-settings-gpu.sh

stage=$1
LNG=$2
REF_LNG=$3
trial_num=$4
eval_on_test=$5

###############################################################################
# Evaluation script for Official Isovec experiments.
#
# by Kelly Marchisio, Apr/May 2022.
###############################################################################

EXP_NAME=real-isovec
OUTDIR=exps/$EXP_NAME/$stage/$trial_num/$LNG-$REF_LNG
SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt
MAPPED_OUTDIR=$OUTDIR/mapped
REF_EMBS=$DIR/exps/baseline/10/$REF_LNG/embs.out
mkdir -p $MAPPED_OUTDIR

if [[ -z "$eval_on_test" ]]; then
	TEST=data/dicts/MUSE/$LNG-$REF_LNG/dev/$LNG-$REF_LNG.6501-8000.txt
elif [[ $eval_on_test == test ]]; then
	TEST=data/dicts/MUSE/$LNG-$REF_LNG/test/$LNG-$REF_LNG.5000-6500.txt
else
	echo Please specify "test" to eval on test set, or nothing to eval on dev.
	exit
fi

set -x
for mode in sup semisup unsup
do
	echo Mapping embeddings with ref embeddings in $mode mode...
    time sh map.sh -s $OUTDIR/embs.out -t $REF_EMBS \
		-u $MAPPED_OUTDIR/embs.out.to$REF_LNG.mapped.$mode \
		-v $MAPPED_OUTDIR/$REF_LNG.mapped.$mode -m $mode -d $SEEDS
	echo Evaluating mapped embeddings on $TEST
	time sh eval.sh -s $MAPPED_OUTDIR/embs.out.to$REF_LNG.mapped.$mode \
		-t $MAPPED_OUTDIR/$REF_LNG.mapped.$mode -d $TEST
done
echo Done.

