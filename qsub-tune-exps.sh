#!/bin/bash

. ./local-settings.sh

stage=$1
LNG=$2
REF_LNG=$3

OUTDIR=$DIR/exps/tune/$stage/$LNG
mkdir -p $OUTDIR

resources='hostname=!c03*&!c21*&!c06*&c*,gpu=1,ram_free=25G'
qsub -l $resources -cwd \
	-o $OUTDIR/qsub.`date +"%Y-%m-%d.%H-%M-%S"`.out \
	-e $OUTDIR/qsub.`date +"%Y-%m-%d.%H-%M-%S"`.err \
	$DIR/tune-exps.sh $stage $LNG $REF_LNG
