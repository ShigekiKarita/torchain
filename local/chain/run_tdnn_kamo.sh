#!/bin/bash
log() {
    echo -e "[LOG] $(date +'%Y-%m-%d %H:%M:%S') $1" >&2
}
max() {
    if [ $1 -ge $2 ];then
        echo $1
    else
        echo $2
    fi
}
min() {
    if [ $1 -le $2 ];then
        echo $1
    else
        echo $2
    fi
}


set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
train_stage=-10
get_egs_stage=-10

decode_iter=
train_set=csj_ntt_headset_arrayA_train
test_sets="ntt2011n_headset_dev ntt2011n_headset_evl ntt2011n_arrayA_dev ntt2011n_arrayA_evl"
gmm=tri4
nnet3_affix=
lang=data/lang_ntt
nj=10

# The rest are configs specific to this script. Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=1a  # affix for the TDNN directory name
tree_affix=

use_fmllr=true
use_ivector=true

# training options
# training chunk-options
decode_iter=
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=10
minibatch_size=128,64
frames_per_eg=150,140,100
remove_egs=true
common_egs_dir=
xent_regularize=0.1

test_online_decoding=false  # if true, it will run the last decoding stage.

convert_noisy_ali_to_clean=false

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

[ $# -ne 0 ] && echo "Usage: $0" && exit 1

if ! cuda-compiled; then
    cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --stop-stage $($use_ivector && echo 7 || echo 3) \
                                  --train-set $train_set \
                                  --test-sets "$test_sets" \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" \
                                  --use-fmllr $use_fmllr \
                                  --nj $nj || exit 1;

gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp${nnet3_affix}
tree_dir=exp/chain${nnet3_affix}/tree${tree_affix:+_$tree_affix}
newlang=data/lang_chain${nnet3_affix}
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp  \
    $lores_train_data_dir/feats.scp; do
    [ ! -f $f ] && log "$0; expected file $f to exist" && exit 1;
done

if [ $stage -le 8 ]; then
  if $use_fmllr; then
      log "$0: aligning with the perturbed low-resolution data using fmllr"
      steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
          data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1;
  else
      log "$0: aligning with the perturbed low-resolution data independently from speaker information"
      steps/align_si.sh --nj $nj --cmd "$train_cmd" \
          data/${train_set}_sp data/lang $gmm_dir $ali_dir || exit 1;
  fi
fi

if [ $stage -le 9 ]; then
    if $use_fmllr;then
        log "$0: aligning using fmllr"
        # Get the alignments as lattices (gives the LF-MMI training more freedom).
        # use the same num-jobs as the alignments
        steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
          data/lang $gmm_dir $lat_dir
        rm $lat_dir/fsts.*.gz # save space
    else
        log "$0: aligning independently from speaker information"
        # Get the alignments as lattices (gives the LF-MMI training more freedom).
        # use the same num-jobs as the alignments
        local/align_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
          data/lang $gmm_dir $lat_dir
        rm $lat_dir/fsts.*.gz # save space
    fi
fi

if [ $stage -le 10 ]; then
    if $convert_noisy_ali_to_clean;then
        log "$0: Try to convery ali of array to headset"
        new_lat=${lat_dir}_clean
        new_ali=${ali_dir}_clean

        mkdir -p $new_lat
        mkdir -p $new_ali

        local/ali_modify/create_array_to_headset_map.py data/${train_set}_sp/segments > $new_lat/convert_table.txt
        local/ali_modify/ali_swap.sh --cmd "$train_cmd" $ali_dir $new_lat/convert_table.txt $new_ali
        local/ali_modify/lat_swap.sh --cmd "$train_cmd" $lat_dir $new_lat/convert_table.txt $new_lat

    fi
fi

if $convert_noisy_ali_to_clean;then
    lat_dir=${lat_dir}_clean
    ali_dir=${ali_dir}_clean
fi

if [ $stage -le 11 ]; then
    # Create a version of the lang/ directory that has one state per phone in the
    # topo file. [note, it really has two states.. the first one is only repeated
    # once, the second one has zero or more repeats.]
    rm -rf $newlang
    cp -r data/lang $newlang
    silphonelist=$(cat $newlang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $newlang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$newlang/topo
fi

if [ $stage -le 12 ]; then
    # Build a tree using our new topology. This is the critically different
    # step compared with other recipes.
    steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
        --leftmost-questions-truncate $leftmost_questions_truncate \
        --context-opts "--context-width=2 --central-position=1" \
        --cmd "$train_cmd" 7000 ${lores_train_data_dir} $newlang $ali_dir $tree_dir
fi


if [ $stage -le 13 ]; then
    log "$0: creating neural net configs using the xconfig parser";

    num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
    learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

    mkdir -p $dir/configs
    if $use_ivector;then
        cat <<EOF > $dir/configs/network.xconfig
input dim=100 name=ivector
input dim=40 name=input

# please note that it is important to have input layer with the name=input
# as the layer immediately preceding the fixed-affine-layer to enable
# the use of short notation for the descriptor
fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

# the first splicing is moved before the lda layer, so no splicing here
relu-batchnorm-layer name=tdnn1 dim=625
relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
relu-batchnorm-layer name=tdnn3 dim=625
relu-batchnorm-layer name=tdnn4 input=Append(-1,0,1) dim=625
relu-batchnorm-layer name=tdnn5 dim=625
relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=625
relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=625
relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=625

## adding the layers for chain branch
relu-batchnorm-layer name=prefinal-chain input=tdnn9 dim=625 target-rms=0.5
output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

# adding the layers for xent branch
# This block prints the configs for a separate output that will be
# trained with a cross-entropy objective in the 'chain' models... this
# has the effect of regularizing the hidden parts of the model.  we use
# 0.5 / args.xent_regularize as the learning rate factor- the factor of
# 0.5 / args.xent_regularize is suitable as it means the xent
# final-layer learns at a rate independent of the regularization
# constant; and the 0.5 was tuned so as to make the relative progress
# similar in the xent and regular final layers.
relu-batchnorm-layer name=prefinal-xent input=tdnn9 dim=625 target-rms=0.5
output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
    else
        cat <<EOF > $dir/configs/network.xconfig
input dim=40 name=input

# please note that it is important to have input layer with the name=input
# as the layer immediately preceding the fixed-affine-layer to enable
# the use of short notation for the descriptor
fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat

# the first splicing is moved before the lda layer, so no splicing here
relu-batchnorm-layer name=tdnn1 dim=625
relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
relu-batchnorm-layer name=tdnn3 dim=625
relu-batchnorm-layer name=tdnn4 input=Append(-1,0,1) dim=625
relu-batchnorm-layer name=tdnn5 dim=625
relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625
relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=625
relu-batchnorm-layer name=tdnn8 input=Append(-3,0,3) dim=625
relu-batchnorm-layer name=tdnn9 input=Append(-3,0,3) dim=625

## adding the layers for chain branch
relu-batchnorm-layer name=prefinal-chain input=tdnn9 dim=625 target-rms=0.5
output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

# adding the layers for xent branch
# This block prints the configs for a separate output that will be
# trained with a cross-entropy objective in the 'chain' models... this
# has the effect of regularizing the hidden parts of the model.  we use
# 0.5 / args.xent_regularize as the learning rate factor- the factor of
# 0.5 / args.xent_regularize is suitable as it means the xent
# final-layer learns at a rate independent of the regularization
# constant; and the 0.5 was tuned so as to make the relative progress
# similar in the xent and regular final layers.
relu-batchnorm-layer name=prefinal-xent input=tdnn9 dim=625 target-rms=0.5
output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
    fi

    steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 14 ]; then
    if $use_ivector;then
        ivec_option="--feat.online-ivector-dir $train_ivector_dir"
    else
        ivec_option=""
    fi

    steps/nnet3/chain/train.py --stage $train_stage \
        --cmd "$decode_cmd" \
        $ivec_option \
        --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
        --chain.xent-regularize $xent_regularize \
        --chain.leaky-hmm-coefficient 0.1 \
        --chain.l2-regularize 0.00005 \
        --chain.apply-deriv-weights false \
        --chain.lm-opts="--num-extra-lm-states=2000" \
        --egs.dir "$common_egs_dir" \
        --egs.stage $get_egs_stage \
        --egs.opts "--frames-overlap-per-eg 0" \
        --egs.chunk-width $frames_per_eg \
        --trainer.num-chunk-per-minibatch $minibatch_size \
        --trainer.frames-per-iter 1500000 \
        --trainer.num-epochs $num_epochs \
        --trainer.optimization.num-jobs-initial $num_jobs_initial \
        --trainer.optimization.num-jobs-final $num_jobs_final \
        --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
        --trainer.optimization.final-effective-lrate $final_effective_lrate \
        --trainer.max-param-change $max_param_change \
        --cleanup.remove-egs $remove_egs \
        --feat-dir $train_data_dir \
        --tree-dir $tree_dir \
        --lat-dir $lat_dir \
        --dir $dir  || exit 1;

fi

if [ $stage -le 15 ]; then
    utils/mkgraph.sh --self-loop-scale 1.0 $lang $dir $dir/graph

    for decode_set in $test_sets; do
        if $use_ivector;then
            ivecdir=exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires
        else
            ivecdir=
        fi

        steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
            --nj $(min $nj $(cat data/${decode_set}_hires/spk2utt | wc -l)) \
            --cmd "$decode_cmd" \
            --online-ivector-dir "$ivecdir" \
            $dir/graph data/${decode_set}_hires $dir/decode_${decode_set} &
        unset ivecdir
    done
fi

if [ $stage -le 16 ]; then
    if $use_ivector;then
        ivecextractor=exp/nnet3${nnet3_affix}/extractor
    else
        ivecextractor=
    fi

    steps/online/nnet3/prepare_online_decoding.sh \
        --mfcc-config conf/mfcc_hires.conf $newlang  $ivecextractor \
        $dir ${dir}_online
    unset ivecextractor

    for decode_set in $test_sets; do
        steps/online/nnet3/decode.sh \
            --nj $(min $nj $(cat data/${decode_set}_hires/spk2utt | wc -l)) \
            --cmd "$decode_cmd" \
            --acwt 1.0 --post-decode-acwt 10.0 \
            $dir/graph data/${decode_set}_hires ${dir}_online/decode_${decode_set} &
    done
fi

wait
