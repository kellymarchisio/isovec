#!/bin/bash

. ./local-settings.sh

stage=$1
model=$2
LNG=$3
REF_LNG=$4

OUTDIR=$DIR/exps/verif/$stage/$LNG/$model
mkdir -p $OUTDIR

resources='hostname=c*,gpu=1,mem_free=25G,ram_free=25G'
qsub -l $resources -cwd \
	-o $OUTDIR/qsub.`date +"%Y-%m-%d.%H-%M-%S"`.out \
	-e $OUTDIR/qsub.`date +"%Y-%m-%d.%H-%M-%S"`.err \
	$DIR/imp-verif.sh $stage $model $LNG $REF_LNG
