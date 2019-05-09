# On PyTorch integration

## requirements

- python 3.7
- pytorch 1.1.0
- cuda 10.0 (recommended)

## how to create test files

- mfcc1.ark

```bash
utils/filter_scp.pl --exclude exp/chain_simple/tdnn1g/egs/valid_uttlist data/train_hires/split100/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/train_hires/split100/1/utt2spk scp:data/train_hires/split100/1/cmvn.scp scp:- ark:- > test/mfcc1.ark
```

- supervision1.ark

```bash
lattice-align-phones --replace-output-symbols=true exp/chain_simple/tri3b_train_lats/final.mdl "ark:gunzip -c exp/chain_simple/tri3b_train_lats/lat.1.gz |" ark:- | chain-get-supervision --lattice-input=true --frame-subsampling-factor=3 --right-tolerance=5 --left-tolerance=5 exp/chain_simple/tdnn1g/tree exp/chain_simple/tdnn1g/0.trans_mdl ark:- ark:- > test/supevision1.ark
```

When I just ran `run.sh`, I learned how to generate these files from this log: `./exp/chain_simple/tdnn1g/egs/log/get_egs.1.log`
```
# lattice-align-phones --replace-output-symbols=true exp/chain_simple/tri3b_train_lats/final.mdl "ark:gunzip -c exp/chain_simple/tri3b_train_lats/lat.1.gz |" ark:- | chain-get-supervision --lattice-input=true --frame-subsampling-factor=3 --right-tolerance=5 --left-tolerance=5 exp/chain_simple/tdnn1g/tree exp/chain_simple/tdnn1g/0.trans_mdl ark:- ark:- | nnet3-chain-get-egs --srand=$[1+0] --left-context=29 --right-context=29 --num-frames=140,100,160 --frame-subsampling-factor=3 --compress=true --num-frames-overlap=0 "ark,s,cs:utils/filter_scp.pl --exclude exp/chain_simple/tdnn1g/egs/valid_uttlist data/train_hires/split100/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/train_hires/split100/1/utt2spk scp:data/train_hires/split100/1/cmvn.scp scp:- ark:- |" ark,s,cs:- ark:- | nnet3-chain-copy-egs --random=true --srand=$[1+0] ark:- ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.1.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.2.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.3.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.4.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.5.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.6.ark 
# Started at Thu May  9 01:19:09 JST 2019
#
nnet3-chain-copy-egs --random=true --srand=1 ark:- ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.1.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.2.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.3.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.4.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.5.ark ark:exp/chain_simple/tdnn1g/egs/cegs_orig.1.6.ark 
chain-get-supervision --lattice-input=true --frame-subsampling-factor=3 --right-tolerance=5 --left-tolerance=5 exp/chain_simple/tdnn1g/tree exp/chain_simple/tdnn1g/0.trans_mdl ark:- ark:- 
nnet3-chain-get-egs --srand=1 --left-context=29 --right-context=29 --num-frames=140,100,160 --frame-subsampling-factor=3 --compress=true --num-frames-overlap=0 'ark,s,cs:utils/filter_scp.pl --exclude exp/chain_simple/tdnn1g/egs/valid_uttlist data/train_hires/split100/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/train_hires/split100/1/utt2spk scp:data/train_hires/split100/1/cmvn.scp scp:- ark:- |' ark,s,cs:- ark:- 
LOG (nnet3-chain-get-egs[5.5.0-76bd]:ComputeDerived():nnet-example-utils.cc:335) Rounding up --num-frames=140,100,160 to multiples of --frame-subsampling-factor=3, to: 141,102,162
lattice-align-phones --replace-output-symbols=true exp/chain_simple/tri3b_train_lats/final.mdl 'ark:gunzip -c exp/chain_simple/tri3b_train_lats/lat.1.gz |' ark:- 
apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/train_hires/split100/1/utt2spk scp:data/train_hires/split100/1/cmvn.scp scp:- ark:- 
LOG (lattice-align-phones[5.5.0-76bd]:main():lattice-align-phones.cc:104) Successfully aligned 610 lattices; 0 had errors.
LOG (apply-cmvn[5.5.0-76bd]:main():apply-cmvn.cc:81) Copied 605 utterances.
LOG (chain-get-supervision[5.5.0-76bd]:main():chain-get-supervision.cc:150) Generated chain supervision information for 610 utterances, errors on 0
LOG (nnet3-chain-get-egs[5.5.0-76bd]:~UtteranceSplitter():nnet-example-utils.cc:357) Split 605 utts, with total length 301485 frames (0.837458 hours assuming 100 frames per second)
LOG (nnet3-chain-get-egs[5.5.0-76bd]:~UtteranceSplitter():nnet-example-utils.cc:366) Average chunk length was 135.603 frames; overlap between adjacent chunks was 1.0299% of input length; length of output was 100.887% of input length (minus overlap = 99.8567%).
LOG (nnet3-chain-get-egs[5.5.0-76bd]:~UtteranceSplitter():nnet-example-utils.cc:382) Output frames are distributed among chunk-sizes as follows: 102 = 16.33%, 141 = 66.2%, 162 = 17.47%
LOG (nnet3-chain-copy-egs[5.5.0-76bd]:main():nnet3-chain-copy-egs.cc:395) Read 2243 neural-network training examples, wrote 2243
# Accounting: time=11 threads=1
# Ended (code 0) at Thu May  9 01:19:20 JST 2019, elapsed time 11 seconds
```
