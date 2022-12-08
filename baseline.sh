#!/bin/bash -v
. ./local-settings.sh

stage=$1
LNG=$2
RAND_SEED=$3

###############################################################################
# Baselines.
#
# by Kelly Marchisio, Apr 2022.
#
# usage: ./baseline.sh stage language random_seed
#
###############################################################################

OUTDIR=exps/baseline/$stage$RAND_SEED/$LNG/
mkdir -p $OUTDIR

DIM=300
ITER=10
WINDOW=5
NEGATIVE=10
MIN_COUNT=10
BATCH_SIZE=16384
STARTING_ALPHA=0.001 # mysg only -- w2v starting alpha defaults to 0.025
INFILE=data/news.2020.$LNG.tok.1M
PRINT_FREQ=100

# Link to training file.
if [ $stage == w2v ]; then
	ln -sf $DIR/$INFILE $OUTDIR/train.tok
    time $WORD2VEC -train $OUTDIR/train.tok -output $OUTDIR/embs.out -cbow 0 \
    	-size $DIM -window $WINDOW -negative $NEGATIVE \
    	-hs 0 -iter 5 -min-count $MIN_COUNT
	exit
elif [ $stage == 1 ]; then
	WARMUP=0.25
	WARMUP_TYPE=percent
	
#####################
# Bigger Data Exps.
#####################
elif [ $stage == w2v-big ]; then
	INFILE=data/news.2018-20.en.tok.full
	ln -sf $DIR/$INFILE $OUTDIR/train.tok
    time $WORD2VEC -train $OUTDIR/train.tok -output $OUTDIR/embs.out -cbow 0 \
    	-size $DIM -window $WINDOW -negative $NEGATIVE \
    	-hs 0 -iter 5 -min-count $MIN_COUNT
	exit

###############################
# Diff Dom (CommonCrawl) Exps.
###############################
elif [ $stage == w2v-cc ]; then #SEND
	INFILE=data/commoncrawl.en.tok
	ln -sf $DIR/$INFILE $OUTDIR/train.tok
    time $WORD2VEC -train $OUTDIR/train.tok -output $OUTDIR/embs.out -cbow 0 \
    	-size $DIM -window $WINDOW -negative $NEGATIVE \
    	-hs 0 -iter 5 -min-count $MIN_COUNT
	exit
else
	echo Stage not recognized. Exiting. && exit
fi

set -x

ln -sf $DIR/$INFILE $OUTDIR/train.tok
time sh train.sh -f $OUTDIR/train.tok -o $OUTDIR -l $LNG -w $WINDOW \
	-n $NEGATIVE -m $MIN_COUNT -d $DIM -i $ITER -b $BATCH_SIZE \
	-h $WARMUP_TYPE -j unsupervised \
	-a $STARTING_ALPHA -k 0 -p Adam -u $WARMUP -q skipgram \
	-r None -s None -v constant -x 1 -y $PRINT_FREQ \
	-z 0 -c $RAND_SEED -g 0 -e 0 #noop
echo Done.

