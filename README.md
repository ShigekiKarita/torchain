# Voxforge recipe for nnet3 chain

This repository contains a tiny but practical recipe for [the chain model](http://kaldi-asr.org/doc/chain.html).
Basically this recipe is derived from `egs/voxforge/s5` but this recipe includes a new stage for the chain model and excludes all the GMM variants (e.g., fMMI, SGMM, etc) not used in existing other `chain` recipes.


## how to use

```
git clone https://github.com/kaldi-asr/kaldi.git
cd kaldi
git checkout 76bdf206f2f988d84c6aa885acbb24c33f05e75b
# install kaldi (see ./INSTALL)
cd egs/voxforge
git clone <this repo>
cd <this repos>
./run.sh
```
