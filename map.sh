#!/bin/bash

###############################################################################
###
### Vecmap map embeddings.
###
###     -- Kelly Marchisio, Feb 2022.

. ./local-settings.sh

# https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
while getopts "d:s:t:u:v:m:" opt; do
  case $opt in
    s) SRC_EMBS="$OPTARG"
    ;;
    t) TRG_EMBS="$OPTARG"
    ;;
    u) SRC_OUT="$OPTARG"
    ;;
    v) TRG_OUT="$OPTARG"
    ;;
    m) MODE="$OPTARG"
    ;;
    d) TRAIN_DICT="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG"
    ;;
  esac
done

echo Performing Vecmap Mapping...
echo SRC_EMBS = $SRC_EMBS
echo TRG_EMBS = $TRG_EMBS
echo SRC_OUT = $SRC_OUT
echo TRG_OUT = $TRG_OUT
echo MODE = $MODE
echo TRAIN_DICT = $TRAIN_DICT

if [ $MODE == 'unsup' ]; then
	python3 $VECMAP/map_embeddings.py --unsupervised \
		--max_embs 200000 -v $SRC_EMBS $TRG_EMBS $SRC_OUT $TRG_OUT --cuda
elif [ $MODE == 'sup' ]; then
	python3 $VECMAP/map_embeddings.py --supervised $TRAIN_DICT \
		--max_embs 200000 -v $SRC_EMBS $TRG_EMBS $SRC_OUT $TRG_OUT --cuda
elif [ $MODE == 'semisup' ]; then
	python3 $VECMAP/map_embeddings.py --semi_supervised $TRAIN_DICT \
		--max_embs 200000 -v $SRC_EMBS $TRG_EMBS $SRC_OUT $TRG_OUT --cuda
else
	echo Invalid Mapping Mode. Exiting. && exit
fi

