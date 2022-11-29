###############################################################################
# Non-Mapped Embeddings
###############################################################################

########################
# Baseline Evals.
########################
for num in 0 1 2 3 4
do
	for lang in bn uk ta
	do
		OUTDIR=exps/baseline/1$num
		SRC_EMBS=$OUTDIR/$lang/embs.out
		REF_EMBS=$OUTDIR/en/embs.out
		for metric in evs rsim gh
		do
			sh qsub-sim-metrics.sh $lang en $SRC_EMBS $REF_EMBS $metric $OUTDIR
		done
	done
done

########################
# Experimental Evals.
########################
for num in 0 1 2 3 4
do
	for lang in bn uk ta
	do
		for exp in 10d 10b-pw 10c-pwi rs10e rs10f-i rs10dU2k evs10bU
		do
			OUTDIR=exps/real-isovec/$exp/$num/$lang-en
			SRC_EMBS=$OUTDIR/embs.out
			REF_EMBS=exps/baseline/10/en/embs.out
			for metric in evs rsim gh
			do
				sh qsub-sim-metrics.sh $lang en $SRC_EMBS $REF_EMBS $metric $OUTDIR
			done
		done
	done
done

###############################################################################
# Mapped Embeddings
###############################################################################

######################################
# Baseline Evals - Mapped in Sup mode.
######################################
for num in 0 1 2 3 4
do
	for lang in bn uk ta
	do
		OUTDIR=exps/baseline/1$num/$lang-en/test
		SRC_EMBS=$OUTDIR/embs.out.toen.mapped.sup
		REF_EMBS=$OUTDIR/en.mapped.sup
		for metric in evs rsim gh
		do
			sh qsub-sim-metrics.sh $lang en $SRC_EMBS $REF_EMBS $metric $OUTDIR
		done
	done
done

######################################
# Baseline Evals - Mapped in Semi-Sup mode.
######################################
for num in 0 1 2 3 4
do
	for lang in bn uk ta
	do
		EXPDIR=exps/baseline/1$num/$lang-en/
		OUTDIR=$EXPDIR/mapped-semisup-iso-metrics
		mkdir -p $OUTDIR
		SRC_EMBS=$EXPDIR/test/embs.out.toen.mapped.semisup
		REF_EMBS=$EXPDIR/test/en.mapped.semisup
		for metric in evs rsim gh
		do
			sh qsub-sim-metrics.sh $lang en $SRC_EMBS $REF_EMBS $metric $OUTDIR
		done
	done
done

########################
# Experimental Evals - Mapped in Semisup mode.
########################
for num in 0 1 2 3 4
do
	for lang in bn uk ta
	do
 		for exp in 10d 10b-pw 10c-pwi # rs10e rs10f-i rs10dU2k evs10bU
 		do
			EXPDIR=exps/real-isovec/$exp/$num/$lang-en
			OUTDIR=$EXPDIR/mapped-semisup-iso-metrics
			mkdir -p $OUTDIR
			SRC_EMBS=$EXPDIR/mapped/embs.out.toen.mapped.semisup
			REF_EMBS=$EXPDIR/mapped/en.mapped.semisup
			for metric in evs rsim gh
			do
				sh qsub-sim-metrics.sh $lang en $SRC_EMBS $REF_EMBS $metric $OUTDIR
			done
		done
	done
done
