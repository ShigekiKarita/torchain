# Voxforge recipe for nnet3 chain

This repository contains a tiny but practical recipe using voxforge and [the chain model](http://kaldi-asr.org/doc/chain.html) for beginners. Basically this recipe is derived from `egs/voxforge/s5` and `egs/wsj/s5` but this recipe includes a new stage for the chain model and excludes all the GMM variants (e.g., fMMI, SGMM, etc) not used in existing other `chain` recipes.


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

## note

### non-default tools deps

- srilm (language model): needs to download .tgz manually from http://www.speech.sri.com/projects/srilm/download.html
- sequitur (grapheme-to-phoneme model): needs to `python setup,py install` manually at `tools/sequitur`

### GMM-HMM acoustic model setting 

First of all, you need to know "Kaldi terms". See http://kaldi-asr.org/doc/glossary.html

misc:
- "alignment" is a HMM predicted (1-best) phoneme for each input frame on one utterance
- "lattice" is a directed acyclic graph (DAG) generating (N-best) recogniton hypotheses per one utterance. See http://kaldi-asr.org/doc/lattices.html
- "tree" is a set of decision tree diagrams for clustering phonemes. See http://kaldi-asr.org/doc/tree_externals.html
- "b" of tri2b has no special meaning. "tri2a" was a kind of "tri2b" without LDA+MLLT but its alignments were not used for subsequent-pass AMs.
- "nosp" refers to the dictionary before silence probabilities and pronunciation probabilities are added
- LDA, MLLT and SAT are feature/speaker adaptation methods
- HCLG.fst (H: HMM, C: context, L: lexicon (dictionary), G: grammer (LM)) http://kaldi-asr.org/doc/graph.html


#### In WSJ

- We refer to `egs/wsj/s5/run.sh` as a baseline recipe for the chain model
- In the WSJ recipe, GMM-HMM models are built as follows (eventually larger feature with larger data through multi-pass alignments):
  - mono: monophone GMM on MFCC 2k short subset of si84 (smaller training set)
  - tri1: triphone GMM on MFCC+deltas half subset of si84 with mono alignments
  - tri2b: triphone GMM on MFCC+deltas LDA+MLLT full si84 with tri1 alignments
  - tri3b: triphone GMM on MFCC+deltas LDA+MLLT+SAT full si284 (larger traning set) with tri2b alignments

#### In Voxforge

Everything is same to WSJ but a pronunciation dictionary (CMUdict+G2P). GMM training may take about 1-2 hour (overall)

- GMM-HMM model stages
  - mono: monophone GMM on MFCC of a 1k train subset
```
Overall, lattice depth (10,50,90-percentile)=(1,23,222) and mean=82.9
steps/diagnostic/analyze_lats.sh: see stats in exp/mono/decode/log/analyze_lattice_depth_stats.log
...
exp/mono/decode/wer_9
%WER 45.21 [ 2019 / 4466, 66 ins, 624 del, 1329 sub ]
%SER 84.80 [ 424 / 500 ]
```
  - tri1: triphone GMM on MFCC+deltas with mono alignments
```
Overall, lattice depth (10,50,90-percentile)=(1,4,51) and mean=21.9
steps/diagnostic/analyze_lats.sh: see stats in exp/tri1/decode/log/analyze_lattice_depth_stats.log
...
exp/tri1/decode/wer_14
%WER 20.96 [ 936 / 4466, 29 ins, 316 del, 591 sub ]
%SER 58.20 [ 291 / 500 ]
```
  - tri2b: triphone GMM on MFCC+deltas and LDA+MLLT with tri1 alignments
```
Overall, lattice depth (10,50,90-percentile)=(1,3,39) and mean=15.3
steps/diagnostic/analyze_lats.sh: see stats in exp/tri2b/decode/log/analyze_lattice_depth_stats.log
...
exp/tri2b/decode/wer_14
%WER 21.20 [ 947 / 4466, 35 ins, 329 del, 583 sub ]
%SER 58.20 [ 291 / 500 ]
```
  - tri3b: triphone GMM on MFCC+deltas and LDA+MLLT+SAT with tri2b alignments
```
Overall, lattice depth (10,50,90-percentile)=(1,2,9) and mean=4.4
steps/diagnostic/analyze_lats.sh: see stats in exp/tri3b/decode/log/analyze_lattice_depth_stats.log
...
exp/tri3b/decode/wer_17
%WER 14.11 [ 630 / 4466, 50 ins, 177 del, 403 sub ]
%SER 50.40 [ 252 / 500 ]
```

### Chain model setting

#### in WSJ

- stages
  - 11. `local/nnet3/run_ivector_common.sh`: apply speed perturbation (only for the train set), 40-dim MFCC (see `conf/mfcc_hires.conf`) and i-vector feature extraction, and alignments from GMM with speed perturbation (only for the train set)
    - input: `data/train (MFCC), data/test (MFCC), exp/tri3b (GMM)`
    - output: `data/train_sp_hires (sp hires MFCC), data/tri3b_train_ali_sp (sp alignments), exp/nnet3/ivectors_train_sp_hires (i-vector), data/test_hires (sp hires MFCC)`
  - 12. `steps/nnet3/chain/gen_topo.py`: create a new HMM topology that has two states (transition or loop) per phone for chain
    - input: `data/lang (pronunciation dictionary)`
    - output: `data/lang_chain`
  - 13. `steps/align_fmllr_lats.sh`: get lattices with GMM and LMs
    - input: `data/train_sp, data/lang data/tri3b`
    - output: `exp/chain/tri3b_train_sp_lats` 
  - 14. `steps/nnet3/chain/build_tree.sh`: create a new tree on the chain HMM topology
    - input: `data/`
  - 

#### in Voxforge

ditto

## debug

### kaldi tools

see http://kaldi-asr.org/doc/tools.html

- show-alignments: `gunzip -c exp/mono_ali/ali.1.gz | show-alignments ./exp/mono_ali/phones.txt ./exp/mono_ali/final.mdl ark:-`
- draw-tree: `draw-tree data/lang/phones.txt exp/mono/tree | dot -Tpng > mono_tree.png`
![mono](mono_tree.png)


## license

BSL-1.0 except for inherited scripts from kaldi
