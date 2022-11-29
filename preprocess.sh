. ./local-settings.sh > /dev/null

INFILE=$1
LNG=$2
THREADS=20

cat $INFILE | $MOSES_SCRIPTS/tokenizer/normalize-punctuation.perl $LNG \
	| $MOSES_SCRIPTS/tokenizer/lowercase.perl \
	| $MOSES_SCRIPTS/tokenizer/remove-non-printing-char.perl \
	| $MOSES_SCRIPTS/tokenizer/tokenizer.perl -a -l $LNG -threads $THREADS
