IsoVec: Controlling the Relative Isomorphism of Word Embedding Spaces
======================

This is an implementation of the experiments and combination system presented
in:
- Kelly Marchisio, Neha Verma, Kevin Duh, Philipp Koehn. 2022. **[IsoVec: Controlling the Relative Isomorphism of Word Embedding Spaces
](https://arxiv.org/abs/2210.05098)**. In *Proceedings of EMNLP 2022*.

If you use this software for academic research, please cite the paper above.

Requirements
--------
- python3
- pytorch
- sklearn
- scipy
- numpy
- indicnlp (for Tamil and Bengali tokenizers)
--------

Setup
-------
- download third party packages: `cd third_party && sh get_third_party.sh && cd ..`
- download and make data: `cd data && sh make_data.sh`

Usage
-------

Baselines:
- `sh baseline.sh {w2v,w2v-big,w2v-cc} {uk,bn,ta} {0,1,2,3,4}`

