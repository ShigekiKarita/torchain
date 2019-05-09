/**
python module to emulate nnet3-chain-get-egs.cc that creates chunk-level feat/supervision/i-vector (NnetChainExample)

TODO(karita): split source code

NOTE: on nnet3-chain-get-egs.cc

Usage:  nnet3-chain-get-egs [options] [<normalization-fst>] <features-rspecifier> <chain-supervision-rspecifier> <egs-wspecifier>


Example: nnet3-chain-get-egs
# [options]
--srand=$[1+0] --left-context=29 --right-context=29 --num-frames=140,100,160 --frame-subsampling-factor=3 --compress=true --num-frames-overlap=0
# <features-rspecifier>
"ark,s,cs:utils/filter_scp.pl --exclude exp/chain_simple/tdnn1g/egs/valid_uttlist data/train_hires/split100/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/train_hires/split100/1/utt2spk scp:data/train_hires/split100/1/cmvn.scp scp:- ark:- |"
# <chain-supervision-rspecifier>
"ark,s,cs:lattice-align-phones --replace-output-symbols=true exp/chain_simple/tri3b_train_lats/final.mdl \"ark:gunzip -c exp/chain_simple/tri3b_train_lats/lat.1.gz |\" ark:- | chain-get-supervision --lattice-input=true --frame-subsampling-factor=3 --right-tolerance=5 --left-tolerance=5 exp/chain_simple/tdnn1g/tree exp/chain_simple/tdnn1g/0.trans_mdl ark:- ark:-"
# <egs-wspecifier>
ark:-

NOTE(karita): Maybe our wrapper takes these essential args at first
[required]
<feat> hi-res MFCC or something
e.g., "ark,s,cs:utils/filter_scp.pl --exclude exp/chain_simple/tdnn1g/egs/valid_uttlist data/train_hires/split100/1/feats.scp | apply-cmvn --norm-means=false --norm-vars=false --utt2spk=ark:data/train_hires/split100/1/utt2spk scp:data/train_hires/split100/1/cmvn.scp scp:- ark:- |"
<supervision> subsampled GMM alignments
e.g., "ark,s,cs:lattice-align-phones --replace-output-symbols=true exp/chain_simple/tri3b_train_lats/final.mdl \"ark:gunzip -c exp/chain_simple/tri3b_train_lats/lat.1.gz |\" ark:- | chain-get-supervision --lattice-input=true --frame-subsampling-factor=3 --right-tolerance=5 --left-tolerance=5 exp/chain_simple/tdnn1g/tree exp/chain_simple/tdnn1g/0.trans_mdl ark:- ark:-"

TODO(karita): support consistent subsampling factor between chain-get-supervision and Config

[optional]
--ivector: TODO(karita)
--normalization-fst: TODO(karita)
--srand: ??
--left-context=29:
--right-context=29:
--num-frames=140,100,160: ??
--frame-subsampling-factor=3:
--left-tolerance=5:
--right-tolerance=5:

Raw log: ./exp/chain_simple/tdnn1g/egs/log/get_egs.1.log
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
 */
#include <sstream>
#include <memory>

#include <torch/extension.h>

// kaldi deps
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-example-utils.h"

using namespace kaldi;
using namespace kaldi::nnet3;

// alias
using Config = kaldi::nnet3::ExampleGenerationConfig;
using Example = kaldi::nnet3::NnetChainExample;


/// helper class derived from nnet3-chain-get-egs.cc
class GetEgs
{
private:
    const std::string feat_rspecifier, supervision_rspecifier,
        normalization_fst_rxfilename, online_ivector_rspecifier;

    fst::StdVectorFst normalization_fst;
    // kaldi resource handlers
    kaldi::RandomAccessBaseFloatMatrixReader feat_reader;
    kaldi::chain::RandomAccessSupervisionReader supervision_reader;
    kaldi::RandomAccessBaseFloatMatrixReader online_ivector_reader;

    // chunk setting
    Config eg_config;
    std::unique_ptr<kaldi::nnet3::UtteranceSplitter> utt_splitter;

public:


    // setup resource
    GetEgs(
        const std::string& feat_rspecifier,
        const std::string& supervision_rspecifier,
        const std::string& normalization_fst_rxfilename,
        const std::string& online_ivector_rspecifier
        )
        : feat_rspecifier(feat_rspecifier),
          supervision_rspecifier(supervision_rspecifier),
          normalization_fst_rxfilename(normalization_fst_rxfilename),
          online_ivector_rspecifier(online_ivector_rspecifier)
     {
         // default
         eg_config.left_context = 29;
         eg_config.right_context = 29;
         eg_config.num_frames_str = "140,100,160";
         eg_config.frame_subsampling_factor = 3;
         eg_config.ComputeDerived();
     }

    bool is_open() const
    {
        return feat_reader.IsOpen() && supervision_reader.IsOpen();
    }

    /// mainly copied from main() in nnet3-chain-get-egs.cc
    void open()
    {
        if (!normalization_fst_rxfilename.empty()) {
            ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);
            KALDI_ASSERT(normalization_fst.NumStates() > 0);

            // TODO(karita) support scaling
            // if (normalization_fst_scale <= 0.0)
            //     KALDI_ERR << "Invalid scale on normalization FST; must be > 0.0";
            // if (normalization_fst_scale != 1.0)
            //     ApplyProbabilityScale(normalization_fst_scale, &normalization_fst);
        }

        // Read as GeneralMatrix so we don't need to un-compress and re-compress
        // when selecting parts of matrices.
        feat_reader.Open(feat_rspecifier);
        supervision_reader.Open(supervision_rspecifier);
        if (!online_ivector_rspecifier.empty())
        {
            online_ivector_reader.Open(online_ivector_rspecifier);
        }
        // TODO(karita) support this
        // RandomAccessBaseFloatVectorReader deriv_weights_reader(
        //     deriv_weights_rspecifier);
    }

    void close()
    {
        if (feat_reader.IsOpen())
        {
            feat_reader.Close();
        }
        if (supervision_reader.IsOpen())
        {
            supervision_reader.Close();
        }
        if (online_ivector_reader.IsOpen())
        {
            online_ivector_reader.Close();
        }
    }

    std::string to_string() const
    {
        std::stringstream ss;
        ss << "torchain.egs.GetEgs(\n"
           << "  feat='" << this->feat_rspecifier << "',\n"
           << "  supervision='" << this->supervision_rspecifier << "',\n"
           << "  normalization_fst='" << this->normalization_fst_rxfilename << "',\n"
           << "  online_ivector='" << this->online_ivector_rspecifier << "'\n)";
        return ss.str();
    }

    void set_config(Config config)
    {
        this->eg_config = config;
        this->eg_config.ComputeDerived();
        this->utt_splitter.reset(new kaldi::nnet3::UtteranceSplitter(this->eg_config));
    }

    Config get_config() const
    {
        return this->eg_config;
    }

    void load(std::string uttid, Example& dst, bool close=true)
    {
        this->open();

        // TODO(karita): copy ProcessFile(...) in nnet3-chain-get-egs.cc here

        if (close) this->close();
    }
};


// NOTE: these pybind11 symbols are imported from <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    py::class_<Config>(m, "Config", "kaldi::nnet3::ExampleGenerationConfig")
        .def(py::init<>())
        .def_readwrite("left_context", &Config::left_context)
        .def_readwrite("right_context", &Config::right_context)
        .def_readwrite("left_context_initial", &Config::left_context_initial)
        .def_readwrite("right_context_final", &Config::right_context_final)
        .def_readwrite("num_frames_overlap", &Config::num_frames_overlap)
        .def_readwrite("frame_subsampling_factor", &Config::frame_subsampling_factor)
        .def_readwrite("num_frames_str", &Config::num_frames_str)
        .def_readwrite("num_frames", &Config::num_frames)
        .def("compute_derived", &Config::ComputeDerived,
             "This function decodes 'num_frames_str' into 'num_frames',\n"
             "and ensures that the members of 'num_frames' are multiples of 'frame_subsampling_factor'.")
        .def("__repr__",
             [](const Config &a) {
                 std::stringstream ss;
                 ss << "torchain.egs.Config(\n"
                    << "  left_context=" << a.left_context << ",\n"
                    << "  right_context=" << a.right_context << ",\n"
                    << "  left_context_initial=" << a.left_context_initial << ",\n"
                    << "  right_context_final=" << a.right_context_final << ",\n"
                    << "  num_frames_overlap=" << a.num_frames_overlap << ",\n"
                    << "  frame_subsampling_factor=" << a.frame_subsampling_factor << ",\n"
                    << "  num_frames_str=" << a.num_frames_str << "\n)";
                 return ss.str();
             }
            );

    py::class_<Example>(m, "Example", "kaldi::nnet3::NnetChainExample")
        .def(py::init());

    py::class_<GetEgs>(m, "GetEgs", "class wrapped kaldi/src/chainbin/nnet3-chain-get-egs.cc")
        .def(py::init<const std::string&, const std::string&, const std::string&, const std::string&>(),
             py::arg("feat"), py::arg("supervision"), py::arg("normalization_fst") = "", py::arg("online_ivector") = "")
        .def("load", &GetEgs::load,
             py::arg("uttid"), py::arg("dst"), py::arg("close") = true)
        .def_property("config", &GetEgs::get_config, &GetEgs::set_config)
        .def_property("config", &GetEgs::get_config, &GetEgs::set_config)
        .def("__repr__", &GetEgs::to_string);
}
