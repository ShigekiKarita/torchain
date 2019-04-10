#!/bin/bash
# -*- sh-basic-offset: 2 -*-
# Copyright 2012 Vassil Panayotov
#           2019 Shigeki Karita
# Apache 2.0

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

. ./path.sh || exit 1

# If you have cluster of machines running GridEngine you may want to
# change the train and decode commands in the file below
. ./cmd.sh || exit 1

# If you rerun this recipe, you can skip some stages you want
stage=0
# Where to save/load dataset
data_root=db

# The number of parallel jobs to be started for some parts of the recipe
# Make sure you have enough resources(CPUs and RAM) to accomodate this number of jobs
njobs=8

# This recipe can select subsets of VoxForge's data based on the "Pronunciation dialect"
# field in VF's etc/README files. To select all dialects, set this to "English"
dialects="((American)|(British)|(Australia)|(Zealand))"

# The number of randomly selected speakers to be put in the test set
nspk_test=20

# Test-time language model order
lm_order=2

# Word position dependent phones?
pos_dep_phones=true

# The directory below will be used to link to a subset of the user directories
# based on various criteria(currently just speaker's accent)
selected=data/selected

# The user of this script could change some of the above parameters. Example:
# /bin/bash run.sh --pos-dep-phones false
. utils/parse_options.sh || exit 1


[[ $# -ge 1 ]] && { echo "Unexpected arguments"; exit 1; }

if [ ${stage} -le 0 ]; then
  echo "Stage 0: Data download"
  if [ -e ${data_root}/extracted ]; then
    echo "${data_root}/extracted already exists. so skip download"
  else
    echo "download data at ${data_root}"
    DATA_ROOT=${data_root} ./getdata.sh
  fi
fi

if [ ${stage} -le 1 ]; then
  echo "Stage 1: Data preparation"
  # Select a subset of the data to use
  # WARNING: the destination directory will be deleted if it already exists!
  local/voxforge_select.sh --dialect $dialects ${data_root}/extracted ${selected} || exit 1

  # Mapping the anonymous speakers to unique IDs
  local/voxforge_map_anonymous.sh ${selected} || exit 1

  # Initial normalization of the data
  local/voxforge_data_prep.sh --nspk_test ${nspk_test} ${selected} || exit 1

  # Prepare ARPA LM and vocabulary using SRILM
  local/voxforge_prepare_lm.sh --order ${lm_order} || exit 1

  # Prepare the lexicon and various phone lists
  # Pronunciations for OOV words are obtained using a pre-trained Sequitur model
  local/voxforge_prepare_dict.sh || exit 1

  # Prepare data/lang and data/local/lang directories
  utils/prepare_lang.sh --position-dependent-phones $pos_dep_phones \
                        data/local/dict '!SIL' data/local/lang data/lang || exit 1

  # Prepare G.fst and data/{train,test} directories
  local/voxforge_format_data.sh || exit 1
fi


if [ ${stage} -le 2 ]; then
  echo "Stage 2: Feature extraction"
  # Now make MFCC features.
  # mfccdir should be some place with a largish disk where you
  # want to store MFCC features.
  mfccdir=data/mfcc
  for x in train test; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $njobs \
                       data/$x exp/make_mfcc/$x $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1;
  done
fi


if [ ${stage} -le 3 ]; then
  echo "Stage 3: Monophone GMM acoustic model (mono)"
  # Train monophone models on a subset of the data
  utils/subset_data_dir.sh data/train 1000 data/train.1k  || exit 1;
  steps/train_mono.sh --nj $njobs --cmd "$train_cmd" data/train.1k data/lang exp/mono || exit 1;

  # Monophone decoding
  utils/mkgraph.sh data/lang_test exp/mono exp/mono/graph || exit 1
  # note: local/decode.sh calls the command line once for each
  # test, and afterwards averages the WERs into (in this case
  # exp/mono/decode/
  steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
                  exp/mono/graph data/test exp/mono/decode

  # Get alignments from monophone system.
  steps/align_si.sh --nj $njobs --cmd "$train_cmd" \
                    data/train data/lang exp/mono exp/mono_ali || exit 1;
fi


if [ ${stage} -le 4 ]; then
  echo "Stage 4: Triphone GMM acoustic model (tri1)"
  # train tri1 [first triphone pass]
  steps/train_deltas.sh --cmd "$train_cmd" \
                        2000 11000 data/train data/lang exp/mono_ali exp/tri1 || exit 1;

  # decode tri1
  utils/mkgraph.sh data/lang_test exp/tri1 exp/tri1/graph || exit 1;
  steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
                  exp/tri1/graph data/test exp/tri1/decode

  #draw-tree data/lang/phones.txt exp/tri1/tree | dot -Tpdf > tree.pdf

  # align tri1
  steps/align_si.sh --nj $njobs --cmd "$train_cmd" \
                    --use-graphs true data/train data/lang exp/tri1 exp/tri1_ali || exit 1;
fi


if [ ${stage} -le 5 ]; then
  echo "Stage 5: Triphone GMM acoustic model for MFCC+deltas LDA+MLLT on tri1 alignments (tri2b)"

  # train and decode tri2b [LDA+MLLT]
  steps/train_lda_mllt.sh --cmd "$train_cmd" 2000 11000 \
                          data/train data/lang exp/tri1_ali exp/tri2b || exit 1;
  utils/mkgraph.sh data/lang_test exp/tri2b exp/tri2b/graph
  steps/decode.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
                  exp/tri2b/graph data/test exp/tri2b/decode

  # Align all data with LDA+MLLT system (tri2b)
  steps/align_si.sh --nj $njobs --cmd "$train_cmd" --use-graphs true \
                    data/train data/lang exp/tri2b exp/tri2b_ali || exit 1;
fi


if [ ${stage} -le 6 ]; then
  echo "Stage 6: Triphone GMM acoustic model for MFCC+deltas and LDA+MLLT+SAT on tri2b alignments (tri3b)"
  ## Do LDA+MLLT+SAT, and decode.
  steps/train_sat.sh 2000 11000 data/train data/lang exp/tri2b_ali exp/tri3b || exit 1;
  utils/mkgraph.sh data/lang_test exp/tri3b exp/tri3b/graph || exit 1;
  steps/decode_fmllr.sh --config conf/decode.config --nj $njobs --cmd "$decode_cmd" \
                        exp/tri3b/graph data/test exp/tri3b/decode || exit 1;


  # Align all data with LDA+MLLT+SAT system (tri3b)
  steps/align_fmllr.sh --nj $njobs --cmd "$train_cmd" --use-graphs true \
                       data/train data/lang exp/tri3b exp/tri3b_ali || exit 1;
fi


./local/chain/run_tdnn.sh --stage $stage --train_set train --test_sets test --gmm tri3b --nj ${njobs} --num_threads_ubm 1 --remove_egs false

