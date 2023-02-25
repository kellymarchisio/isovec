#!/bin/bash

. ./local-settings.sh

stage=$1
SRC=$2
TRG=$3
eval_on_test=$4

OUTDIR=$DIR/exps/baseline/$stage/$SRC-$TRG/$eval_on_test
mkdir -p $OUTDIR

resources='hostname=!c14*&!c13*&!c21*&c*,gpu=1,mem_free=10G,ram_free=10G'
qsub -l $resources -cwd \
	-o $OUTDIR/qsub$eval_on_test.`date +"%Y-%m-%d.%H-%M-%S"`.out \
	-e $OUTDIR/qsub$eval_on_test.`date +"%Y-%m-%d.%H-%M-%S"`.err \
	$DIR/map-baseline.sh $stage $SRC $TRG $eval_on_test
