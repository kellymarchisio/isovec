#!/bin/bash

. ./local-settings.sh

SRC=$1
TRG=$2
SRC_EMBS=$3
TRG_EMBS=$4
MODE=$5
OUTDIR=$6

if [ $MODE == 'gh' ]; then
	resources='hostname=c*|b1[12345679]*,gpu=1,mem_free=10G,ram_free=10G'
else
	resources='hostname=c*|b1[12345679]*,mem_free=10G,ram_free=10G'
fi

qsub -l $resources -cwd \
	-o $OUTDIR/qsub-metrics.$MODE.$SRC-$TRG.`date +"%Y-%m-%d.%H-%M-%S"`.out \
	-e $OUTDIR/qsub-metrics.$MODE.$SRC-$TRG.`date +"%Y-%m-%d.%H-%M-%S"`.err \
	./sim-metrics.sh $SRC $TRG $SRC_EMBS $TRG_EMBS $MODE
