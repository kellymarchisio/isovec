#!/bin/bash -v
. ./local-settings.sh

MODE=$1
STAGE=$2
SRC=$3
TRG=$4
EVAL=$5
SEEDS=$DIR/data/dicts/$SRC-$TRG/train/$SRC-$TRG.0-5000.txt

# Choose dev/test set.
if [[ $EVAL == dev ]]; then
	TEST=data/dicts/$SRC-$TRG/dev/$SRC-$TRG.6501-8000.txt
elif [[ $EVAL == test ]]; then
	TEST=data/dicts/$SRC-$TRG/test/$SRC-$TRG.5000-6500.txt
else
	echo Please specify "test" to eval on test set, or "dev" to eval on dev set.
	exit
fi

# Set up mapping directory and point to correct embeddings.
if [[ $MODE == baseline ]]; then
	if ! ([[ $STAGE == w2v ]] || [[ $STAGE == isovec ]]); then
		echo Stage must be w2v or isovec for baseline evaluation.
		exit
	fi
	BASEDIR=$DIR/exps/baseline/$STAGE # isovec or w2v
	SRC_EMBS=$BASEDIR/$SRC/embs.out
	TRG_EMBS=$BASEDIR/$TRG/embs.out
	MAPPED_OUTDIR=$BASEDIR/$SRC-$TRG/$EVAL
elif [[ $MODE == isovec ]]; then
	if ! ([[ $STAGE == l2 ]] || \
		[[ $STAGE == proc-l2 ]] || [[ $STAGE == proc-l2-init ]] || \
		[[ $STAGE == rsim ]] || [[ $STAGE == rsim-init ]] || \
		[[ $STAGE == rsim-u ]] || [[ $STAGE == evs-u ]]); then
		echo Please specify a correct stage for isovec
		exit
	fi
	BASEDIR=$DIR/exps/isovec/$STAGE
	SRC_EMBS=$BASEDIR/$SRC-$TRG/embs.out
	TRG_EMBS=$DIR/exps/baseline/isovec/$TRG/embs.out # Reference embeddings
	MAPPED_OUTDIR=$BASEDIR/$SRC-$TRG/$EVAL
else
	echo Please specify "isovec" or "baseline" for mode.
	exit
fi

echo Source Embeddings $SRC_EMBS
echo Reference Embeddings $TRG_EMBS
echo Output directory: $MAPPED_OUTDIR

# Perform the VecMap mapping and eval.
mkdir -p $MAPPED_OUTDIR
set -x
for mode in sup semisup unsup
do
	echo Mapping embeddings with ref embeddings in $mode mode...
    time sh map.sh -s $SRC_EMBS -t $TRG_EMBS \
		-u $MAPPED_OUTDIR/embs.out.to$TRG.mapped.$mode \
		-v $MAPPED_OUTDIR/$TRG.mapped.$mode -m $mode -d $SEEDS \
		> $MAPPED_OUTDIR/$mode.map-eval.out
	echo Evaluating mapped embeddings...
	time sh eval.sh -s $MAPPED_OUTDIR/embs.out.to$TRG.mapped.$mode \
		-t $MAPPED_OUTDIR/$TRG.mapped.$mode -d $TEST \
		>> $MAPPED_OUTDIR/$mode.map-eval.out
done
echo Done.

