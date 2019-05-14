#!/bin/bash

# author: Shigeki Karita <shigekikarita@gmail.com>

set -e -o pipefail

outdir=test/res

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# gmm=tri3b
# gmm_dir=exp/${gmm}
# ali_dir=exp/${gmm}_ali
# lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_lats
# train_data_dir=data/${train_set}_hires
# train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_hires
# lores_train_data_dir=data/${train_set}

tmpdir=.torchain

echo "generating MFCC subset"
utils/subset_data_dir.sh --first data/train 10 $tmpdir/feat_hires
apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:$tmpdir/utt2spk scp:$tmpdir/cmvn.scp scp:$tmpdir/feats.scp ark:- | subset-feats --n 10 > $outdir/mfcc.ark

echo "generating supervision on subset"
utils/subset_data_dir.sh --first data/train 10 $tmpdir/feat
steps/align_fmllr_lats.sh --nj 1 --cmd "$train_cmd" $tmpdir/feat \
                          data/lang exp/tri3b $tmpdir/lat

lattice-align-phones --replace-output-symbols=true exp/chain_simple/tri3b_train_lats/final.mdl "ark:gunzip -c $tmpdir/lat/lat.1.gz |" ark:- | chain-get-supervision --lattice-input=true --frame-subsampling-factor=3 --right-tolerance=5 --left-tolerance=5 exp/chain_simple/tdnn1g/tree exp/chain_simple/tdnn1g/0.trans_mdl ark:- ark:- > $outdir/supevision.ark

rm -rf $tmpdir
