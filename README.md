IsoVec: Controlling the Relative Isomorphism of Word Embedding Spaces
======================

This is an implementation of the experiments and combination system presented
in:
- Kelly Marchisio, Neha Verma, Kevin Duh, and Philipp Koehn. 2022. **[IsoVec:
  Controlling the Relative Isomorphism of Word Embedding
  Spaces](https://aclanthology.org/2022.emnlp-main.404/)**. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 6019–6033, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

If you use this software for academic research, please cite the paper above.

Requirements
--------
- python3
- pytorch
- sklearn
- scipy
- numpy
- indic-nlp-library 
- torchtext
--------

Setup
-------
- Download third party packages: `cd third_party && sh get_third_party.sh && cd ..`
    * Note: If you're on Mac with an M1 chip, word2vec might not build.  You can fix
    this by changing -march=native to **[-mcpu=apple-m1](https://stackoverflow.com/questions/65966969/why-does-march-native-not-work-on-apple-m1)**
    in word2vec's makefile, and subbing in **[getc\_unlocked and putc\_unlocked](https://github.com/tmikolov/word2vec/pull/40)** for fgetc\_unlocked/fputc\_unlocked.
- Download and make data: `cd data && sh make_data.sh`

Usage
-------
To reproduce Table 1 in the paper (Baselines), run:
- `sh baseline.sh $system $lang $seed`
    * For instance, run `sh baseline.sh w2v uk 0` for offical word2vec trained on Ukrainian.
    * system choices: {isovec, w2v}
    * lang choices: {uk, bn, ta, en}
