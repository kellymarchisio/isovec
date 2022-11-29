#!/bin/bash

. ./local-settings.sh

stage=$1
LNG=$2
REF_LNG=$3
trial_num=$4
eval_on_test=$5

OUTDIR=$DIR/exps/real-isovec/$stage/$trial_num/$LNG-$REF_LNG
mkdir -p $OUTDIR

resources='hostname=!c08*&!c07*&!c04*&!c25*&c*,gpu=1,ram_free=25G'
## resources='hostname=!c08*&!c17*&!c13*&!c04*&!c14*&!c03*&!c21*&c*,gpu=1,ram_free=10G'
qsub -l $resources -cwd \
	-o $OUTDIR/qsub$eval_on_test.`date +"%Y-%m-%d.%H-%M-%S"`.out \
	-e $OUTDIR/qsub$eval_on_test.`date +"%Y-%m-%d.%H-%M-%S"`.err \
	$DIR/run-isovec-eval.sh $stage $LNG $REF_LNG $trial_num $eval_on_test
