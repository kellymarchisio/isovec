#!/bin/bash -v
. ./local-settings.sh

stage=$1
LNG=$2
REF_LNG=$3
trial_num=$4

###############################################################################
# Combining Skipgram & Isomorphism Losses
#
# by Kelly Marchisio, Apr 2022.
###############################################################################

EXP_NAME=real-isovec
OUTDIR=exps/$EXP_NAME/$stage/$trial_num/$LNG-$REF_LNG
SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt
TEST=data/dicts/MUSE/$LNG-$REF_LNG/dev/$LNG-$REF_LNG.6501-8000.txt
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
REF_EMBS=$DIR/exps/baseline/10/$REF_LNG/embs.out
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

if [ $stage == 10a ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
elif [ $stage == 10b ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
elif [ $stage == 10c ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
elif [ $stage == 10d ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
elif [ $stage == 10e ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
elif [ $stage == 10f ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
elif [ $stage == 10g ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.0001
elif [ $stage == 10h ]; then
	MIXED_LOSS_START_BATCH=0
	BETA=0.05
elif [ $stage == 10a-pw ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
elif [ $stage == 10b-pw ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
elif [ $stage == 10c-pw ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
elif [ $stage == 10d-pw ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
elif [ $stage == 10e-pw ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
elif [ $stage == 10a-pwi ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	INIT_EMBS_W_REFS=1
elif [ $stage == 10b-pwi ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	INIT_EMBS_W_REFS=1
elif [ $stage == 10c-pwi ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	INIT_EMBS_W_REFS=1
elif [ $stage == 10d-pwi ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	INIT_EMBS_W_REFS=1
elif [ $stage == 10e-pwi ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	INIT_EMBS_W_REFS=1
#######################################################3
# RSim Exps.
#######################################################3
elif [ $stage == rs0 ]; then
	LOSS=rs
	INFILE=data/news.2020.$LNG.tok.1k
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
elif [ $stage == rs10a ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
elif [ $stage == rs10b ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
elif [ $stage == rs10c ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
elif [ $stage == rs10d ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
elif [ $stage == rs10e ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
elif [ $stage == rs10f ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
elif [ $stage == rs10a-i ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	INIT_EMBS_W_REFS=1
elif [ $stage == rs10b-i ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	INIT_EMBS_W_REFS=1
elif [ $stage == rs10c-i ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	INIT_EMBS_W_REFS=1
elif [ $stage == rs10d-i ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	INIT_EMBS_W_REFS=1
elif [ $stage == rs10e-i ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	INIT_EMBS_W_REFS=1
elif [ $stage == rs10f-i ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	INIT_EMBS_W_REFS=1
elif [ $stage == rs0U ]; then
	LOSS=rs
	INFILE=data/news.2020.$LNG.tok.1k
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	MODE=unsupervised
	GH_N=100
elif [ $stage == rs10eU0 ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	MODE=unsupervised
	GH_N=5000
elif [ $stage == rs10aU ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10bU ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10cU ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10dU ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10eU ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	MODE=unsupervised
	GH_N=10000 #OOM at 20k
elif [ $stage == rs10fU ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10aU1k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	MODE=unsupervised
	GH_N=1000
elif [ $stage == rs10bU1k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	MODE=unsupervised
	GH_N=1000
elif [ $stage == rs10cU1k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	MODE=unsupervised
	GH_N=1000
elif [ $stage == rs10dU1k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=1000
elif [ $stage == rs10eU1k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	MODE=unsupervised
	GH_N=1000 #OOM at 20k
elif [ $stage == rs10fU1k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	MODE=unsupervised
	GH_N=1000
elif [ $stage == rs10aU2k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	MODE=unsupervised
	GH_N=2000
elif [ $stage == rs10bU2k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	MODE=unsupervised
	GH_N=2000
elif [ $stage == rs10cU2k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	MODE=unsupervised
	GH_N=2000
elif [ $stage == rs10dU2k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=2000
elif [ $stage == rs10eU2k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	MODE=unsupervised
	GH_N=2000 #OOM at 20k
elif [ $stage == rs10fU2k ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	MODE=unsupervised
	GH_N=2000
elif [ $stage == rs10aU-late ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=5000
	BETA=0.5
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10bU-late ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=5000
	BETA=0.333
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10cU-late ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=5000
	BETA=0.2
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10dU-late ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=5000
	BETA=0.1
	MODE=unsupervised
	GH_N=10000
elif [ $stage == rs10eU-late ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=5000
	BETA=0.01
	MODE=unsupervised
	GH_N=10000 #OOM at 20k
elif [ $stage == rs10fU-late ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=5000
	BETA=0.001
	MODE=unsupervised
	GH_N=10000
#######################################################
# EVS Exps.
#######################################################
elif [ $stage == evs10d0 ]; then
	LOSS=evs
	INFILE=data/news.2020.$LNG.tok.1k
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
elif [ $stage == evs10b ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
elif [ $stage == evs10d ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
elif [ $stage == evs10f ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
elif [ $stage == evs10b-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10d-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10f-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10a2k ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	MAX_SEEDS=2000
elif [ $stage == evs10b2k ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	MAX_SEEDS=2000
elif [ $stage == evs10c2k ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	MAX_SEEDS=2000
elif [ $stage == evs10d2k ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MAX_SEEDS=2000
elif [ $stage == evs10e2k ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	MAX_SEEDS=2000
elif [ $stage == evs10f2k ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	MAX_SEEDS=2000
elif [ $stage == evs10a2k-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	MAX_SEEDS=2000
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10b2k-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	MAX_SEEDS=2000
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10c2k-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	MAX_SEEDS=2000
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10d2k-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MAX_SEEDS=2000
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10e2k-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	MAX_SEEDS=2000
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10f2k-1to1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	MAX_SEEDS=2000
	SEEDS=data/dicts/MUSE/$LNG-$REF_LNG/train/$LNG-$REF_LNG.0-5000.txt.1to1
elif [ $stage == evs10dU0 ]; then
	LOSS=evs
	INFILE=data/news.2020.$LNG.tok.1k
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=100
elif [ $stage == evs10aU ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.5
	MODE=unsupervised
	GH_N=2000
elif [ $stage == evs10bU ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.333
	MODE=unsupervised
	GH_N=2000
elif [ $stage == evs10cU ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	MODE=unsupervised
	GH_N=2000
elif [ $stage == evs10dU1 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=1000
elif [ $stage == evs10dU ] || [ $stage == evs10dU2 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=2000
elif [ $stage == evs10dU4 ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=4000
elif [ $stage == evs10eU ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.01
	MODE=unsupervised
	GH_N=2000 #OOM at 20k
elif [ $stage == evs10fU ]; then
	LOSS=evs
	MIXED_LOSS_START_BATCH=0
	BETA=0.001
	MODE=unsupervised
	GH_N=2000
#######################################################
# GH Exps.
#######################################################
elif [ $stage == gh00 ]; then
	LOSS=gh
	INFILE=data/news.2020.$LNG.tok.1k
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	GH_N=100
elif [ $stage == gh0 ]; then
	LOSS=gh
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	GH_N=1000 # This doesn't matter in supervised mode (which this is)
	PRINT_FREQ=1
	MAX_SEEDS=1000
elif [ $stage == gh1 ]; then
	LOSS=gh
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	GH_N=1000 # This doesn't matter in supervised mode (which this is)
	PRINT_FREQ=1
	MAX_SEEDS=2000
elif [ $stage == gh2 ]; then
	LOSS=gh
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	GH_N=1000 # This doesn't matter in supervised mode (which this is)
	PRINT_FREQ=1
	MAX_SEEDS=4000
#######################################################
# Diff Dom Exps, Diff Alg.
#######################################################
elif [ $stage == 10c-pwi-cc ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	INIT_EMBS_W_REFS=1
	REF_EMBS=$DIR/exps/baseline/w2v-cc0/$REF_LNG/embs.out
elif [ $stage == rs10dU2k-cc ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=2000
	REF_EMBS=$DIR/exps/baseline/w2v-cc0/$REF_LNG/embs.out
#######################################################
# Big En Exps, Diff Alg.
#######################################################
elif [ $stage == 10c-pwi-big ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	INIT_EMBS_W_REFS=1
	REF_EMBS=$DIR/exps/baseline/w2v-big0/$REF_LNG/embs.out
elif [ $stage == rs10dU2k-big ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=2000
	REF_EMBS=$DIR/exps/baseline/w2v-big0/$REF_LNG/embs.out
#######################################################
# Diff Alg Exps.
#######################################################
elif [ $stage == 10c-pwi-w2v ]; then
	LOSS=procwass
	MIXED_LOSS_START_BATCH=0
	BETA=0.2
	INIT_EMBS_W_REFS=1
	REF_EMBS=$DIR/exps/baseline/w2v0/$REF_LNG/embs.out
elif [ $stage == rs10dU2k-w2v ]; then
	LOSS=rs
	MIXED_LOSS_START_BATCH=0
	BETA=0.1
	MODE=unsupervised
	GH_N=2000
	REF_EMBS=$DIR/exps/baseline/w2v0/$REF_LNG/embs.out
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

