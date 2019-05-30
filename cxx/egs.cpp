/**
python module to emulate nnet3-chain-get-egs.cc that creates chunk-level feat/supervision/i-vector (NnetChainExample)
 */

#include <sstream>
#include <memory>
#include <unordered_map>

// third party
#include <torch/extension.h>

// kaldi
#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "hmm/transition-model.h"
#include "hmm/posterior.h"
#include "nnet3/nnet-example.h"
#include "nnet3/nnet-chain-example.h"
#include "nnet3/nnet-example-utils.h"

#include "./tensor.hpp"

using namespace kaldi;
using namespace kaldi::nnet3;


std::string supervison_to_string(const chain::Supervision &a) {
    std::stringstream ss;
    ss << "chain::Supervision(\n"
       << "  label_dim=" << a.label_dim << ",\n"
       << "  num_sequences=" << a.num_sequences << ",\n"
       << "  frames_per_sequence=" << a.frames_per_sequence << ",\n"
       << "  weight=" << a.weight << "\n)";
    return ss.str();
}


struct TorchainExample
{
    std::string key;
    std::unordered_map<std::string, chain::Supervision> outputs; // e.g., GMM phone alignments
    std::unordered_map<std::string, torch::Tensor> inputs; // e.g.,  mfcc and i-vector

    TorchainExample(const NnetChainExample& eg, const std::string& key)
        : key(key)
    {
        // TODO(karita): should we support multiple output?
        for (auto&& o : eg.outputs)
        {
            this->outputs[o.name] = o.supervision;
        }

        // convert kaldi::GeneralMatrix to torch::Tensor
        for (auto&& i : eg.inputs)
        {
            this->inputs[i.name] = torchain::copy_tensor(i.features);
        }
    }

    std::string to_string() const {
        std::stringstream ss;
        ss << "TorchainExample(\n"
           << "  key=" << key << "\n"
           << "  inputs={";
        for (const auto& i : this->inputs) {
            ss << "'" << i.first << "': Tensor(shape=" << i.second.sizes() << "), ";
        }
        ss << "}\n";
        ss << "  outputs={";
        for (const auto& o : this->outputs) {
            ss << "'" << o.first << "': " << supervison_to_string(o.second) << "), ";
        }
        ss << "}\n";
        ss  << ")";
        return ss.str();
    }
};


/**
   This function does all the processing for one utterance, and outputs the
   supervision objects to 'example_writer'.

     @param [in]  trans_mdl           The transition-model for the tree for which we
                                      are dumping egs.  This is expected to be
                                      NULL if the input examples already contain
                                      pdfs-ids+1 in their FSTs, and non-NULL if the
                                      input examples contain transition-ids in
                                      their FSTs and need to be converted to
                                      unconstrained 'e2e' (end-to-end) style FSTs
                                      which contain pdf-ids+1 but which won't enforce any
                                      alignment constraints interior to the
                                      utterance.
     @param [in]  normalization_fst   A version of denominator FST used to add weights
                                      to the created supervision. It is
                                      actually an FST expected to have the
                                      labels as (pdf-id+1).  If this has no states,
                                      we skip the final stage of egs preparation
                                      in which we compose with the normalization
                                      FST, and you should do it later with
                                      nnet3-chain-normalize-egs.
     @param [in]  feats               Input feature matrix
     @param [in]  ivector_feats       Online iVector matrix sub-sampled at a
                                      rate of "ivector_period".
                                      If NULL, iVector will not be added
                                      as in input to the egs.
     @param [in]  ivector_period      Number of frames between iVectors in
                                      "ivector_feats" matrix.
     @param [in]  supervision         Supervision for 'chain' training created
                                      from the binary chain-get-supervision.
                                      This is expected to be at a
                                      sub-sampled rate if
                                      --frame-subsampling-factor > 1.
     @param [in]  deriv_weights       Vector of per-frame weights that scale
                                      a frame's gradient during backpropagation.
                                      If NULL, this is equivalent to specifying
                                      a vector of all 1s.
                                      The dimension of the vector is expected
                                      to be the supervision size, which is
                                      at a sub-sampled rate if
                                      --frame-subsampling-factor > 1.
     @param [in]  supervision_length_tolerance
                                      Tolerance for difference in num-frames-subsampled between
                                      supervision and deriv weights, and also between supervision
                                      and input frames.
     @param [in]  utt_id              Utterance-id
     @param [in]  compress            If true, compresses the feature matrices.
     @param [out]  utt_splitter       Pointer to UtteranceSplitter object,
                                      which helps to split an utterance into
                                      chunks. This also stores some stats.
     @param [out]  example_writer     cleared at first. then chunk level examples are pushed.

**/

static bool ProcessFile(const TransitionModel *trans_mdl,
                        const fst::StdVectorFst &normalization_fst,
                        const GeneralMatrix &feats,
                        const MatrixBase<BaseFloat> *ivector_feats,
                        int32 ivector_period,
                        const chain::Supervision &supervision,
                        const VectorBase<BaseFloat> *deriv_weights,
                        int32 supervision_length_tolerance,
                        const std::string &utt_id,
                        bool compress,
                        UtteranceSplitter *utt_splitter,
                        std::vector<TorchainExample>& example_writer) {
  KALDI_ASSERT(supervision.num_sequences == 1);
  int32 num_input_frames = feats.NumRows(),
      num_output_frames = supervision.frames_per_sequence;

  int32 frame_subsampling_factor = utt_splitter->Config().frame_subsampling_factor;

  if (deriv_weights && (std::abs(deriv_weights->Dim() - num_output_frames)
                        > supervision_length_tolerance)) {
    KALDI_WARN << "For utterance " << utt_id
               << ", mismatch between deriv-weights dim and num-output-frames"
               << "; " << deriv_weights->Dim() << " vs " << num_output_frames;
    return false;
  }

  if (!utt_splitter->LengthsMatch(utt_id, num_input_frames, num_output_frames,
                                  supervision_length_tolerance))
    return false;  // LengthsMatch() will have printed a warning.

  // It can happen if people mess with the feature frame-width options, that
  // there can be small mismatches in length between the supervisions (derived
  // from lattices) and the features; if this happens, and
  // supervision_length_tolerance is nonzero, and the num-input-frames is larger
  // than plausible for this num_output_frames, then it could lead us to try to
  // access frames in the supervision that don't exist.  The following
  // if-statement is to prevent that happening.
  if (num_input_frames > num_output_frames * frame_subsampling_factor)
    num_input_frames = num_output_frames * frame_subsampling_factor;

  std::vector<ChunkTimeInfo> chunks;

  utt_splitter->GetChunksForUtterance(num_input_frames, &chunks);

  if (chunks.empty()) {
    KALDI_WARN << "Not producing egs for utterance " << utt_id
               << " because it is too short: "
               << num_input_frames << " frames.";
    return false;
  }

  chain::SupervisionSplitter sup_splitter(supervision);

  example_writer.clear();
  for (size_t c = 0; c < chunks.size(); c++) {
    ChunkTimeInfo &chunk = chunks[c];

    int32 start_frame_subsampled = chunk.first_frame / frame_subsampling_factor,
        num_frames_subsampled = chunk.num_frames / frame_subsampling_factor;

    chain::Supervision supervision_part;
    sup_splitter.GetFrameRange(start_frame_subsampled,
                               num_frames_subsampled,
                               &supervision_part);

    if (trans_mdl != NULL)
      ConvertSupervisionToUnconstrained(*trans_mdl, &supervision_part);

    if (normalization_fst.NumStates() > 0 &&
        !AddWeightToSupervisionFst(normalization_fst,
                                   &supervision_part)) {
      KALDI_WARN << "For utterance " << utt_id << ", feature frames "
                 << chunk.first_frame << " to "
                 << (chunk.first_frame + chunk.num_frames)
                 << ", FST was empty after composing with normalization FST. "
                 << "This should be extremely rare (a few per corpus, at most)";
    }

    int32 first_frame = 0;  // we shift the time-indexes of all these parts so
                            // that the supervised part starts from frame 0.

    NnetChainExample nnet_chain_eg;
    nnet_chain_eg.outputs.resize(1);

    SubVector<BaseFloat> output_weights(
        &(chunk.output_weights[0]),
        static_cast<int32>(chunk.output_weights.size()));

    if (!deriv_weights) {
      NnetChainSupervision nnet_supervision("output", supervision_part,
                                            output_weights,
                                            first_frame,
                                            frame_subsampling_factor);
      nnet_chain_eg.outputs[0].Swap(&nnet_supervision);
    } else {
      Vector<BaseFloat> this_deriv_weights(num_frames_subsampled);
      for (int32 i = 0; i < num_frames_subsampled; i++) {
        int32 t = i + start_frame_subsampled;
        if (t < deriv_weights->Dim())
          this_deriv_weights(i) = (*deriv_weights)(t);
      }
      KALDI_ASSERT(output_weights.Dim() == num_frames_subsampled);
      this_deriv_weights.MulElements(output_weights);
      NnetChainSupervision nnet_supervision("output", supervision_part,
                                            this_deriv_weights,
                                            first_frame,
                                            frame_subsampling_factor);
      nnet_chain_eg.outputs[0].Swap(&nnet_supervision);
    }

    nnet_chain_eg.inputs.resize(ivector_feats != NULL ? 2 : 1);

    int32 tot_input_frames = chunk.left_context + chunk.num_frames +
        chunk.right_context,
        start_frame = chunk.first_frame - chunk.left_context;

    GeneralMatrix input_frames;
    ExtractRowRangeWithPadding(feats, start_frame, tot_input_frames,
                               &input_frames);

    NnetIo input_io("input", -chunk.left_context, input_frames);
    nnet_chain_eg.inputs[0].Swap(&input_io);

    if (ivector_feats != NULL) {
      // if applicable, add the iVector feature.
      // choose iVector from a random frame in the chunk
      int32 ivector_frame = RandInt(start_frame,
                                    start_frame + num_input_frames - 1),
          ivector_frame_subsampled = ivector_frame / ivector_period;
      if (ivector_frame_subsampled < 0)
        ivector_frame_subsampled = 0;
      if (ivector_frame_subsampled >= ivector_feats->NumRows())
        ivector_frame_subsampled = ivector_feats->NumRows() - 1;
      Matrix<BaseFloat> ivector(1, ivector_feats->NumCols());
      ivector.Row(0).CopyFromVec(ivector_feats->Row(ivector_frame_subsampled));
      NnetIo ivector_io("ivector", 0, ivector);
      nnet_chain_eg.inputs[1].Swap(&ivector_io);
    }

    if (compress)
      nnet_chain_eg.Compress();

    std::ostringstream os;
    os << utt_id << "-" << chunk.first_frame;

    std::string key = os.str(); // key is <utt_id>-<frame_id>

    // example_writer->Write(key, nnet_chain_eg);
    example_writer.emplace_back(nnet_chain_eg, key);
  }
  return true;
}


// alias
using Config = kaldi::nnet3::ExampleGenerationConfig;
using Example = kaldi::nnet3::NnetChainExample;

// http://kaldi-asr.org/doc/classkaldi_1_1SequentialTableReader.html
struct SeqTensorGenerator : public kaldi::SequentialGeneralMatrixReader {
    SeqTensorGenerator()
        : kaldi::SequentialGeneralMatrixReader() {}

    SeqTensorGenerator(const std::string& rspec)
        : kaldi::SequentialGeneralMatrixReader(rspec) {}

    torch::Tensor tensor() {
        return torchain::copy_tensor(this->Value());
    }
};


/// TODO(karita): add option not to log in nnet-example-utils.cc
/// helper class derived from nnet3-chain-get-egs.cc
struct GetEgs
{
    const std::string feat_rspecifier, supervision_rspecifier,
        normalization_fst_rxfilename, online_ivector_rspecifier,
        deriv_weights_rspecifier,
        trans_mdl_rxfilename;

    // kaldi resource handlers
    kaldi::RandomAccessGeneralMatrixReader feat_reader;
    kaldi::chain::RandomAccessSupervisionReader supervision_reader;
    kaldi::RandomAccessBaseFloatMatrixReader online_ivector_reader;
    kaldi::RandomAccessBaseFloatVectorReader deriv_weights_reader;

    std::unique_ptr<TransitionModel> trans_mdl_ptr;
    fst::StdVectorFst normalization_fst;

    // chunk setting
    Config eg_config;
    std::unique_ptr<kaldi::nnet3::UtteranceSplitter> utt_splitter;

    // options outside ExampleGenerationConfig
    // TODO(karita): bind this readwrite attr
    bool compress = true;
    int32 length_tolerance = 100;
    int32 online_ivector_period = 1;
    int32 supervision_length_tolerance = 1;
    BaseFloat normalization_fst_scale = 1.0;
    int32 srand_seed = 0;


    // setup resource
    GetEgs(
        const std::string& feat_rspecifier,
        const std::string& supervision_rspecifier,
        const std::string& normalization_fst_rxfilename,
        const std::string& online_ivector_rspecifier,
        const std::string& deriv_weights_rspecifier,
        const std::string& trans_mdl_rxfilename
        )
        : feat_rspecifier(feat_rspecifier),
          supervision_rspecifier(supervision_rspecifier),
          normalization_fst_rxfilename(normalization_fst_rxfilename),
          online_ivector_rspecifier(online_ivector_rspecifier),
          deriv_weights_rspecifier(deriv_weights_rspecifier),
          trans_mdl_rxfilename(trans_mdl_rxfilename)
     {
         // default config
         Config config;
         config.left_context = 29;
         config.right_context = 29;
         config.num_frames_str = "140,100,160";
         config.frame_subsampling_factor = 3;
         config.num_frames_overlap = 0;
         this->set_config(config);
     }

    bool is_open() const
    {
        return feat_reader.IsOpen() && supervision_reader.IsOpen();
    }

    /// mainly copied from main() in nnet3-chain-get-egs.cc
    void open()
    {
        if (!normalization_fst_rxfilename.empty())
        {
            ReadFstKaldi(normalization_fst_rxfilename, &normalization_fst);
            KALDI_ASSERT(normalization_fst.NumStates() > 0);

            if (normalization_fst_scale <= 0.0)
                KALDI_ERR << "Invalid scale on normalization FST; must be > 0.0";
            if (normalization_fst_scale != 1.0)
                ApplyProbabilityScale(normalization_fst_scale, &normalization_fst);
        }
        if (!trans_mdl_rxfilename.empty())
        {
            this->trans_mdl_ptr.reset(new TransitionModel);
            ReadKaldiObject(trans_mdl_rxfilename, trans_mdl_ptr.get());
        }

        // Read as GeneralMatrix so we don't need to un-compress and re-compress
        // when selecting parts of matrices.
        feat_reader.Open(feat_rspecifier);
        supervision_reader.Open(supervision_rspecifier);
        if (!online_ivector_rspecifier.empty())
        {
            online_ivector_reader.Open(online_ivector_rspecifier);
        }
        if (!deriv_weights_rspecifier.empty())
        {
            deriv_weights_reader.Open(deriv_weights_rspecifier);
        }
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
        if (deriv_weights_reader.IsOpen())
        {
            deriv_weights_reader.Close();
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

    /// return false only if data loading failed
    std::vector<TorchainExample> load(const std::string& key, bool close=true)
    {
        std::vector<TorchainExample> dst;
        if (!this->is_open()) this->open();

        if (!feat_reader.HasKey(key))
        {
            KALDI_WARN << "No feature for key " << key;
            return {};
        }
        const auto& feats = feat_reader.Value(key);

        if (!supervision_reader.HasKey(key))
        {
            KALDI_WARN << "No pdf-level posterior for key " << key;
            return {};
        }
        else
        {
            const chain::Supervision &supervision = supervision_reader.Value(key);
            const Matrix<BaseFloat> *online_ivector_feats = nullptr;
            if (!online_ivector_rspecifier.empty())
            {
                if (!online_ivector_reader.HasKey(key))
                {
                    KALDI_WARN << "No iVectors for utterance " << key;
                    return {};
                }
                else
                {
                    // this address will be valid until we call HasKey() or Value()
                    // again.
                    online_ivector_feats = &(online_ivector_reader.Value(key));
                }
            }
            if (online_ivector_feats != nullptr &&
                (abs(feats.NumRows() - (online_ivector_feats->NumRows() *
                                        online_ivector_period)) > length_tolerance
                 || online_ivector_feats->NumRows() == 0)) {
                KALDI_WARN << "Length difference between feats " << feats.NumRows()
                           << " and iVectors " << online_ivector_feats->NumRows()
                           << "exceeds tolerance " << length_tolerance;
                return {};
            }

            const Vector<BaseFloat> *deriv_weights = nullptr;
            if (!deriv_weights_rspecifier.empty())
            {
                if (!deriv_weights_reader.HasKey(key)) {
                    KALDI_WARN << "No deriv weights for utterance " << key;
                    return {};
                }
                else
                {
                    // this address will be valid until we call HasKey() or Value()
                    // again.
                    deriv_weights = &(deriv_weights_reader.Value(key));
                }
            }

            if (!ProcessFile(trans_mdl_ptr.get(), normalization_fst, feats,
                             online_ivector_feats, online_ivector_period,
                             supervision, deriv_weights, supervision_length_tolerance,
                             key, compress,
                             utt_splitter.get(), dst))
            {
                return {};
            }
            // TODO(karita) RAII?
        }
        if (close) this->close();
        return dst;
    }
};

std::unique_ptr<chain::Supervision> merge_supervison(const std::vector<const chain::Supervision*> &input) {
    auto ret = std::make_unique<chain::Supervision>();
    chain::MergeSupervision(input, ret.get());
    return ret;
}


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

    // TODO(karita): implement __repr__
    py::class_<TorchainExample>(m, "TorchainExample")
        .def_readwrite("key", &TorchainExample::key)
        .def_readwrite("inputs", &TorchainExample::inputs)
        .def_readwrite("outputs", &TorchainExample::outputs)
        .def("__repr__", &TorchainExample::to_string);

    py::class_<chain::Supervision>(m, "Supervision", "kaldi::chain::Supervision")
        .def_readwrite("weight", &chain::Supervision::weight)
        .def_readwrite("num_sequences", &chain::Supervision::num_sequences)
        .def_readwrite("frames_per_sequence", &chain::Supervision::frames_per_sequence)
        .def_readwrite("label_dim", &chain::Supervision::label_dim)
        .def_readwrite("fst", &chain::Supervision::fst)
        .def("__repr__", &supervison_to_string);

    py::class_<fst::StdVectorFst>(m, "StdVectorFst", "fst::StdVectorFst")
        .def(py::init<>())
        .def("write", py::overload_cast<const std::string&>(&fst::StdVectorFst::Write, py::const_));

    // m.def("read_stdfst", py::overload_cast<const std::string&>(&fst::StdVectorFst::Read));

    py::class_<SeqTensorGenerator>(m, "SeqTensorGenerator")
        .def(py::init<>())
        .def(py::init<const std::string&>(), py::arg("rspecifier"))
        .def("key", &SeqTensorGenerator::Key)
        .def("tensor", &SeqTensorGenerator::tensor)
        .def("next", &SeqTensorGenerator::Next)
        .def("done", &SeqTensorGenerator::Done)
        .def("open", &SeqTensorGenerator::Open)
        .def("is_open", &SeqTensorGenerator::IsOpen)
        .def("close", &SeqTensorGenerator::Close);

    py::class_<GetEgs>(m, "_GetEgs", "class wrapped kaldi/src/chainbin/nnet3-chain-get-egs.cc")
        .def(py::init<const std::string&, const std::string&, const std::string&, const std::string&, const std::string&, const std::string&>(),
             py::arg("feat"),
             py::arg("supervision"),
             py::arg("normalization_fst") = "",
             py::arg("online_ivector") = "",
             py::arg("deriv_weights") = "",
             py::arg("trans_mdl") = ""
             )
        .def("load", &GetEgs::load,
             py::arg("uttid"), py::arg("close") = true)
        .def_property("config", &GetEgs::get_config, &GetEgs::set_config)
        .def_readonly("rspec", &GetEgs::feat_rspecifier)
        .def("__repr__", &GetEgs::to_string);

    m.def("merge_supervison", &merge_supervison);
}
