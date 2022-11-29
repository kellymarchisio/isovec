#!/bin/bash

. ./local-settings.sh

src_stage=$1
ref_stage=$2
SRC=$3
TRG=$4
eval_on_test=$5

OUTDIR=$DIR/exps/baseline/$src_stage/$ref_stage/$SRC-$TRG/$eval_on_test
mkdir -p $OUTDIR

resources='hostname=!c14*&!c13*&!c21*&c*,gpu=1,mem_free=10G,ram_free=10G'
qsub -l $resources -cwd \
	-o $OUTDIR/qsub$eval_on_test.`date +"%Y-%m-%d.%H-%M-%S"`.out \
	-e $OUTDIR/qsub$eval_on_test.`date +"%Y-%m-%d.%H-%M-%S"`.err \
	$DIR/map-baseline-diffembs.sh $src_stage $ref_stage $SRC $TRG $eval_on_test
