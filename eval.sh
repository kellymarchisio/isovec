#!/bin/bash

###############################################################################
###
### Vecmap map embeddings.
###
###     -- Kelly Marchisio, Feb 2022.

. ./local-settings.sh

# https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts
while getopts "s:t:d:" opt; do
  case $opt in
    s) SRC_EMBS="$OPTARG"
    ;;
    t) TRG_EMBS="$OPTARG"
    ;;
    d) TEST_SET="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG"
    ;;
  esac
done

python3 $VECMAP/eval_translation.py $SRC_EMBS $TRG_EMBS \
	-d $TEST_SET --retrieval csls --cuda
