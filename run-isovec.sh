#!/bin/bash -v
. ./local-settings.sh

###############################################################################
#
# Combining Skipgram & Isomorphism Losses
#   by Kelly Marchisio.
#
###############################################################################

STAGE=$1
LNG=$2
REF_LNG=$3

EXP_NAME=isovec
OUTDIR=exps/$EXP_NAME/$STAGE/$LNG-$REF_LNG
SEEDS=data/dicts/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt
TEST=data/dicts/$LNG-$REF_LNG/dev/$LNG-$REF_LNG.6501-8000.txt
MAPPED_OUTDIR=$OUTDIR/mapped
mkdir -p $OUTDIR

DIM=300
ITER=10
WINDOW=5
NEGATIVE=10
MIN_COUNT=10
BATCH_SIZE=16384
WARMUP=0.25
WARMUP_TYPE=percent
STARTING_ALPHA=0.001
INFILE=data/news.2020.$LNG.tok.1M
REF_EMBS=$DIR/exps/baseline/isovec/$REF_LNG/embs.out
RAND_SEED=0 # To match with en space.
LOSS=wass
MODE=supervised
OPT=Adam
PRINT_FREQ=100
BETA=0.5
MIXED_LOSS_START_BATCH=0
BETA_MODE=constant
INIT_EMBS_W_REFS=0
GH_N=10000
MAX_SEEDS=-1 # All.

if [ $STAGE == l2 ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
elif [ $STAGE == proc-l2 ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
elif [ $STAGE == proc-l2-init ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	INIT_EMBS_W_REFS=1
elif [ $STAGE == rsim ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
elif [ $STAGE == rsim-init ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	INIT_EMBS_W_REFS=1
elif [ $STAGE == rsim-u ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=2000
elif [ $STAGE == evs-u ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	MODE=unsupervised
	GH_N=2000
else
	echo Stage not recognized. Exiting. && exit
fi


set -x
# Link to training file.
ln -sf $DIR/$INFILE $OUTDIR/train.tok

time sh train.sh -f $OUTDIR/train.tok -o $OUTDIR \
	-l $LNG -w $WINDOW -c $RAND_SEED -e $MAX_SEEDS \
	-n $NEGATIVE -m $MIN_COUNT -d $DIM -i $ITER -b $BATCH_SIZE \
	-a $STARTING_ALPHA -k $INIT_EMBS_W_REFS -p $OPT -u $WARMUP -q $LOSS \
	-r $REF_EMBS -s $SEEDS -v $BETA_MODE -x $BETA -y $PRINT_FREQ \
	-z $MIXED_LOSS_START_BATCH -j $MODE -h $WARMUP_TYPE -g $GH_N

mkdir -p $MAPPED_OUTDIR
for mode in sup semisup unsup
do
	echo Mapping embeddings with ref embeddings in $mode mode...
    time sh map.sh -s $OUTDIR/embs.out -t $REF_EMBS \
		-u $MAPPED_OUTDIR/embs.out.to$REF_LNG.mapped.$mode \
		-v $MAPPED_OUTDIR/$REF_LNG.mapped.$mode -m $mode -d $SEEDS
	echo Evaluating mapped embeddings...
	time sh eval.sh -s $MAPPED_OUTDIR/embs.out.to$REF_LNG.mapped.$mode \
		-t $MAPPED_OUTDIR/$REF_LNG.mapped.$mode -d $TEST
done
echo Done.

