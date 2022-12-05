#!/bin/bash -v

###############################################################################
###
### Train word embeddings.
###
### This script has been written for use on the JHU CLSP Grid
###     -- Kelly Marchisio, Feb 2022.

. ./local-settings.sh

# https://unix.stackexchange.com/questions/129391/passing-named-arguments-to-shell-scripts

while getopts "a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:u:v:w:x:y:z:" opt; do
  case $opt in
    f) INFILE="$OPTARG"
    ;;
    o) OUTDIR="$OPTARG"
    ;;
    c) RAND_SEED="$OPTARG"
    ;;
    r) REF_EMBS="$OPTARG"
    ;;
    e) MAX_SEEDS="$OPTARG"
    ;;
    s) SEEDS="$OPTARG"
    ;;
    g) GH_N="$OPTARG"
    ;;
    l) LNG="$OPTARG"
    ;;
    w) WINDOW="$OPTARG"
    ;;
    n) NEGATIVE="$OPTARG"
    ;;
    m) MIN_COUNT="$OPTARG"
    ;;
    d) DIM="$OPTARG"
    ;;
    i) ITER="$OPTARG"
    ;;
    a) ALPHA="$OPTARG"
    ;;
    b) BATCH_SIZE="$OPTARG"
    ;;
    p) OPT="$OPTARG"
    ;;
    k) INIT_EMBS_W_REFS="$OPTARG"
    ;;
    u) WARMUP="$OPTARG"
    ;;
    h) WARMUP_TYPE="$OPTARG"
    ;;
    v) BETA_MODE="$OPTARG"
    ;;
    q) LOSS="$OPTARG"
    ;;
    x) BETA="$OPTARG"
    ;;
    y) PRINT_FREQ="$OPTARG"
    ;;
    z) MIXED_LOSS_START_BATCH="$OPTARG"
    ;;
    j) MODE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG"
    ;;
  esac
done

set -x
python3 src/train.py $INFILE $OUTDIR --lang $LNG \
	--window $WINDOW --iters $ITER --warmup $WARMUP \
	--batch $BATCH_SIZE --size $DIM --min-count $MIN_COUNT \
	--negative $NEGATIVE --starting-alpha $ALPHA --opt $OPT --loss $LOSS \
	--ref-embs $REF_EMBS --seeds $SEEDS --mixed-loss-start-batch $MIXED_LOSS_START_BATCH \
	--print-freq $PRINT_FREQ --beta $BETA --beta-mode $BETA_MODE --gh-n $GH_N \
    --init-embs-w-refs $INIT_EMBS_W_REFS --mode $MODE \
	--warmup-type $WARMUP_TYPE --rand-seed $RAND_SEED --max-seeds $MAX_SEEDS
