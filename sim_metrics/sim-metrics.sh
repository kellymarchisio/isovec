#!/bin/bash

. ../local-settings.sh

SRC=$1
TRG=$2
SRC_EMBS=$3
TRG_EMBS=$4
MODE=$5
set -x
if [ $MODE == 'evs' ]; then
	echo Scoring EVS
    python2 $ISOSTUDY_SCRIPTS/evs_script.py $SRC_EMBS $TRG_EMBS
elif [ $MODE == 'gh' ]; then
	echo Scoring GH
    python $ISOSTUDY_SCRIPTS/gh_script.py $SRC_EMBS $TRG_EMBS
elif [ $MODE == 'rsim' ]; then
	echo Scoring rsim
	SEEDS=data/dicts/MUSE/$SRC-$TRG/train/$SRC-$TRG.0-5000.txt.1to1
    python2 $ISOSTUDY_SCRIPTS/rsim_script.py $SRC_EMBS $TRG_EMBS $SEEDS
fi
