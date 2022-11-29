#!/bin/bash -v
. ./local-settings-gpu.sh

stage=$1
SRC=$2
TRG=$3
eval_on_test=$4

SRC_EMBS=exps/baseline/$stage/$SRC/embs.out
TRG_EMBS=exps/baseline/$stage/$TRG/embs.out
MAPPED_OUTDIR=exps/baseline/$stage/$SRC-$TRG/$eval_on_test

SEEDS=data/dicts/MUSE/$SRC-$TRG/train/$SRC-$TRG.0-5000.txt
if [[ -z "$eval_on_test" ]]; then
	TEST=data/dicts/MUSE/$SRC-$TRG/dev/$SRC-$TRG.6501-8000.txt
elif [[ $eval_on_test == test ]]; then
	TEST=data/dicts/MUSE/$SRC-$TRG/test/$SRC-$TRG.5000-6500.txt
else
	echo Please specify "test" to eval on test set, or nothing to eval on dev.
	exit
fi

mkdir -p $MAPPED_OUTDIR
set -x
for mode in sup semisup unsup
do
	echo Mapping embeddings with ref embeddings in $mode mode...
    time sh map.sh -s $SRC_EMBS -t $TRG_EMBS \
		-u $MAPPED_OUTDIR/embs.out.to$TRG.mapped.$mode \
		-v $MAPPED_OUTDIR/$TRG.mapped.$mode -m $mode -d $SEEDS
	echo Evaluating mapped embeddings...
	time sh eval.sh -s $MAPPED_OUTDIR/embs.out.to$TRG.mapped.$mode \
		-t $MAPPED_OUTDIR/$TRG.mapped.$mode -d $TEST
done
echo Done.

