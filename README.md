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
    in word2vec's makefile, and subbing in **[getc\_unlocked and putc\_unlocked](https://github.com/tmikolov/word2vec/pull/40)** for
    fgetc\_unlocked/fputc\_unlocked. You'll also need to use gshuf instead of
    shuf within src/train.py.
- Download and make data: `cd data && sh make_data.sh`
- Download and make train/dev/test dictionaries: `cd data/dicts && sh
  create_dicts.sh`

Usage
-------
To reproduce Table 1 in the paper (Baselines), run:
- `sh baseline.sh $system $lang $seed`
    * For instance, run `sh baseline.sh w2v uk 0` for offical word2vec trained on Ukrainian.
    * system choices: {isovec, w2v}
    * lang choices: {uk, bn, ta, en}
- Here is an example experiment for running Isovec:
    * Goal: Train a Ukrainian embedding space with RSIM-U, in reference to a fixed English space.
    * Step 1: Train the fixed English space with `sh baseline.sh isovec uk 0`
    * Step 2: Train the Ukrainian space with: `sh run-isovec.sh rsim-u uk en 0`
- Choices of Isovec training algorithm are `l2, proc-l2, proc-l2-init, rsim,
  rsim-init, rsim-u, evs-u` for L2, Proc-L2, Proc-L2+Init, RSIM, RSIM-U, and
  EVS-U as detailed in Section 4.3 and 4.4 of the paper. 

