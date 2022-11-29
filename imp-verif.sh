#!/bin/bash -v
. ./local-settings.sh

stage=$1
model=$2
LNG=$3
REF_LNG=$4

###############################################################################
# Sanity-Check - Verifying that our Skip-Gram implementation is reasonable.
#
# 0) Create baseline de vectors with word2vec
# 1) Create en vectors with word2vec or our implementation of skipgram, with
# 	varying data sizes.
# 2) Map the vectors to the baseline de vectors using vecmap
# 3) Evaluate BLI performance.
#
# by Kelly Marchisio, Feb 2022.
###############################################################################

OUTDIR=exps/verif/$stage/$LNG/$model
REF_EMBS=exps/verif/$stage/$REF_LNG/$model/embs.out
TEST=data/dicts/MUSE/$LNG-$REF_LNG/dev/$LNG-$REF_LNG.6501-8000.txt
MAPPED_OUTDIR=$OUTDIR/mapped
mkdir -p $OUTDIR

DIM=300
ITER=5
WINDOW=5
NEGATIVE=10
MIN_COUNT=10
BATCH_SIZE=1
WARMUP=4000
STARTING_ALPHA=0.025
INFILE=data/news.2020.$LNG.tok.1M
OPT=SGD


if [ $stage == 00 ]; then
	INFILE=data/news.2020.$LNG.normtok.lc.200
	BATCH_SIZE=10
fi
if [ $stage == 0 ]; then
	echo Running with defaults.
	# (Results, early Feb 2022: Acc: 26.1 for word2vec with sample=0
	# Called Exp 4 at that time (en) in exps/baseline,
	# matching to ref-de2/w2v [all names changed])
fi
if [ $stage == 0b ]; then
	BATCH_SIZE=64
	STARTING_ALPHA=0.2
fi
if [ $stage == 0c ]; then
	BATCH_SIZE=128
	STARTING_ALPHA=0.25
fi
if [ $stage == 0d ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=0.8
fi
if [ $stage == 0e ]; then
	BATCH_SIZE=4096
	STARTING_ALPHA=1.6
fi
if [ $stage == 0f ]; then
	BATCH_SIZE=128
fi
if [ $stage == 0g ]; then
	BATCH_SIZE=4096
fi
if [ $stage == 0h ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=1.0
fi
if [ $stage == 0i ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=2.0
fi
if [ $stage == 0j ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=5.0
fi
if [ $stage == 0k ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=10.0
	# 5 March: Loss 0.5. train time = 2.5h
fi
if [ $stage == 0l ]; then
	BATCH_SIZE=1024
fi
if [ $stage == 0m ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=25.0
	# word distance looks good. Loss was like 0.54
	# Accuracy:  3.75% for de-en with nn search (loss was 0.57)
	# Accuracy:  3.37% for de-en with csls search.
	# 3.90% with de-en csls search w/ neg_samples fixed.
	# Accuracy:  2.91% for en-de csls search with neg/samples fixed.
	# Loss 0.54-0.57

	# 5 March: Loss 0.5. Haven't done Accuracy yet.
fi
if [ $stage == 0n ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=50.0
	# 4 March
	# 	word distance looks good. Loss was like 0.52
	# 	vecmap mapping accuracy was 7.42% de-en with nn search in eval.
	# 	Accuracy: 5.60% en-de with csls after neg_samples fixed on en side
	# 	but not de side, 7.88% de-en (same - neg samples_fixed only on en)
	# 5 March, after changing embedding initialization.
	# 	De En loss: 0.47, En De loss: 0.49
	#	 Accuracy: 19.25% for en-de - much better! Still not at ~27% like
	# 	   w2v, but we'll work on that.
	#	 Accuracy: 20.35% de-en. was Accuracy: 28.84%
	#	Train time = 3h
	# 7 March, after remaking training data to align with w2v (undersample
	#   target and context words).
	#	De En loss: 0.49, En De loss: 0.50:
	#	 Accuracy: 22.69% en-de
	#	 Accuracy: 24.71% de-en.
fi
if [ $stage == 0n-shuf ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=50.0
fi
if [ $stage == 0o ]; then
	BATCH_SIZE=4096
	STARTING_ALPHA=200.0
	# Avg loss 0.52. Train time: 3h
	# Acc: 5.6% w/ neg_samples on en side fixed (not not de side)
fi
if [ $stage == 0p ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=50.0
	ITER=15
fi
if [ $stage == 0q ]; then
	BATCH_SIZE=4096
	STARTING_ALPHA=100.0
	# Avg loss 0.54. Train time: 3h

	# 5 March: 0.49 loss, train time: 3h
fi
if [ $stage == 0r ]; then
	BATCH_SIZE=4096
	STARTING_ALPHA=1000.0
	ITER=15 #2nd run. First was ITER=5
fi
if [ $stage == 0s ]; then
	BATCH_SIZE=8192
	STARTING_ALPHA=200.0
	# Avg loss 0.54. Train time: 3h
fi
if [ $stage == 1 ]; then
	LNG=en
	REF_LNG=de
	# About 5 hours to create all that data.
	INFILE=data/news.2020.$LNG.normtok.lc.full
	BATCH_SIZE=1024
	STARTING_ALPHA=2.5
	# (Results,early Feb 2022: Coverage: 96.93%  Accuracy: 53.03% for w2v.
	# Called Exp 5 at that time in exps/baseline, matching to ref-de4/w2v
	# [all names changed] That had sample=0.
fi

# These are experiments with Adam. I'm trying to find a reasonable batch-size &
# LR to make the final loss << 0.52. Note, when these were run, we didn't have
# LR scheduling implemented for Adam.
if [ $stage == 0bA ]; then
	BATCH_SIZE=64
	OPT=Adam
	# Loss going up to 60s. Killed.
fi
if [ $stage == 0cA ]; then
	BATCH_SIZE=128
	OPT=Adam
	# Loss going up in 40s. Killed.
fi
if [ $stage == 0dA ]; then
	BATCH_SIZE=1024
	OPT=Adam
	# Loss went up to 13.
fi
if [ $stage == 0eA ]; then
	BATCH_SIZE=4096
	OPT=Adam
	# Loss went up to 5.
fi
if [ $stage == 0fA ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=0.001
	OPT=Adam
	# Loss is 0.56
fi
if [ $stage == 0gA ]; then
	BATCH_SIZE=1024
	STARTING_ALPHA=0.0001
	OPT=Adam
	# Loss is 0.71
fi

# To see if coverage improves with indic tokenizer with only mincount_2 (with
# min_count=10, coverage was 58%)
if [ $stage == 2 ]; then
	MIN_COUNT=2
fi


# Link to training file.
## ln -sf $DIR/$INFILE $OUTDIR/train.tok
##
## set -x
## if [ $model == 'w2v' ]; then
##      time $WORD2VEC -train $OUTDIR/train.tok -output $EMBS_OUT -cbow 0 \
##      	-size $DIM -window $WINDOW -negative $NEGATIVE \
##      	-hs 0 -iter $ITER -min-count $MIN_COUNT
## elif [ $model == 'mysg' ]; then
## 	time sh train.sh -f $OUTDIR/train.tok -o $OUTDIR -l $LNG -w $WINDOW \
## 		-n $NEGATIVE -m $MIN_COUNT -d $DIM -i $ITER -b $BATCH_SIZE \
## 		-a $STARTING_ALPHA -p $OPT -u $WARMUP
## else
## 	echo Invalid word embedding training model. Exiting. && exit
## fi


echo Mapping embeddings with ref embeddings...
mkdir -p $MAPPED_OUTDIR
time sh map.sh -s $OUTDIR/embs.out -t $REF_EMBS \
	-u $MAPPED_OUTDIR/embs.out.to$REF_LNG.mapped  \
	-v $MAPPED_OUTDIR/$REF_LNG.mapped

echo Evaluating...
time sh eval.sh -s $MAPPED_OUTDIR/embs.out.to$REF_LNG.mapped \
	-t $MAPPED_OUTDIR/$REF_LNG.mapped -d $TEST
echo Done.
