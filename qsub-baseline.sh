#!/bin/bash

. ./local-settings.sh

stage=$1
LNG=$2
RAND_SEED=$3

OUTDIR=$DIR/exps/baseline/$stage$RAND_SEED/$LNG
mkdir -p $OUTDIR

if [[ "$stage" == *w2v* ]]; then
	resources='hostname=c*'
else
	resources='hostname=c*,gpu=1,mem_free=25G,ram_free=25G'
fi
qsub -l $resources -cwd \
	-o $OUTDIR/qsub.`date +"%Y-%m-%d.%H-%M-%S"`.out \
	-e $OUTDIR/qsub.`date +"%Y-%m-%d.%H-%M-%S"`.err \
	$DIR/baseline.sh $stage $LNG $RAND_SEED
