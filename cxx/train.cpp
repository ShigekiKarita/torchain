#include <cmath>

// third party
#include <torch/extension.h>
#include <THC/THC.h>

// kaldi
#include "./cu-device-patched.h"
#include "chain/chain-training.h"

#include "./tensor.hpp"


extern "C" {
    // this symbol will be resolved automatically from PyTorch libs
    extern THCState *state;
}

void set_kaldi_device(torch::Tensor t) {
    // NOTE: need to patch kaldi for making all the private fields
    //       e.g., active_gpu_id_ public
    auto gpu_id = t.device().index();
    auto& device = kaldi::CuDevice::this_thread_device_;
    device.initialized_ = true;
    // if (device.device_id_ == gpu_id) return; // maybe no need to reset
    // kaldi::CuDevice::Instantiate().AllowMultithreading();
    device.device_id_ = gpu_id;
    device.device_id_copy_ = gpu_id;
    // device.multi_threaded_ = true; // TODO(karita)
    device.cublas_handle_ = THCState_getCurrentBlasHandle(state);
    device.cusparse_handle_ = THCState_getCurrentSparseHandle(state);
    // device.properties_ = *THCState_getCurrentDeviceProperties(state);
    KALDI_ASSERT(kaldi::CuDevice::Instantiate().Enabled());
}


kaldi::chain::DenominatorGraph
denominator_graph(
    const std::string& rxfilename,
    std::int64_t num_pdfs
    )
{
    fst::StdVectorFst den_fst;
    fst::ReadFstKaldi(rxfilename, &den_fst);
    return kaldi::chain::DenominatorGraph(den_fst, num_pdfs);
}

struct TorchainResult
{
    BaseFloat objf = NAN;
    BaseFloat l2_term = NAN;
    BaseFloat weight = NAN;
    BaseFloat xent = NAN;
};

TorchainResult chain_loss(
    // inputs
    const kaldi::chain::DenominatorGraph& den_graph,
    const kaldi::chain::Supervision& supervision,
    torch::Tensor nnet_output_tensor, // CUDA
    // grads CUDA
    torch::Tensor nnet_output_deriv_tensor,
    torch::Tensor xent_output_deriv_tensor,
    // hyper params
    const kaldi::chain::ChainTrainingOptions& opts)
{
    TorchainResult result;
    cudaDeviceSynchronize();
    set_kaldi_device(nnet_output_tensor);
    auto nnet_output = torchain::make_cusubmatrix(nnet_output_tensor);
    auto nnet_output_deriv = torchain::make_cusubmatrix(nnet_output_deriv_tensor);

    kaldi::CuMatrix<BaseFloat> xent_deriv;
    auto xent_deriv_ptr = opts.xent_regularize != 0.0 ? &xent_deriv : nullptr;
    kaldi::chain::ComputeChainObjfAndDeriv(opts, den_graph, supervision, nnet_output,
                                           &result.objf, &result.l2_term, &result.weight,
                                           &nnet_output_deriv, xent_deriv_ptr);
    if (opts.xent_regularize != 0.0)
    {
        xent_output_deriv_tensor.copy_(torchain::ref_tensor(xent_deriv));
    }
    return result;
}


// NOTE: these pybind11 symbols are imported from <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) 
{
    // TODO(karita): docstring

    py::class_<kaldi::chain::DenominatorGraph>(m, "DenominatorGraph", "kaldi::chain::DenominatorGraph")
        .def("num_states", &kaldi::chain::DenominatorGraph::NumStates)
        .def("num_pdfs", &kaldi::chain::DenominatorGraph::NumPdfs);

    m.def("denominator_graph", &denominator_graph, "load kaldi::chain::DenominatorGraph");

    py::class_<TorchainResult>(m, "TorchainResult")
        .def_readwrite("objf", &TorchainResult::objf)
        .def_readwrite("l2_term", &TorchainResult::l2_term)
        .def_readwrite("weight", &TorchainResult::weight)
        .def_readwrite("xent", &TorchainResult::xent);

    py::class_<kaldi::chain::ChainTrainingOptions>(m, "ChainTrainingOptions")
        .def(py::init<>())
        .def_readwrite("l2_regularize",
                       &kaldi::chain::ChainTrainingOptions::l2_regularize,
                       "l2 regularization "
                       "constant for 'chain' training, applied to the output "
                       "of the neural net.")
        .def_readwrite("leaky_hmm_coefficient",
                       &kaldi::chain::ChainTrainingOptions::leaky_hmm_coefficient, "Coefficient "
                       "that allows transitions from each HMM state to each other "
                       "HMM state, to ensure gradual forgetting of context (can "
                       "improve generalization).  For numerical reasons, may not be "
                       "exactly zero.")
        .def_readwrite("xent_regularize",
                       &kaldi::chain::ChainTrainingOptions::xent_regularize,
                       "Cross-entropy "
                       "regularization constant for 'chain' training.  If "
                       "nonzero, the network is expected to have an output "
                       "named 'output-xent', which should have a softmax as "
                       "its final nonlinearity.");


    // TODO(karita): keyword args
    m.def("_chain_loss_impl", &chain_loss, "wrapper of kaldi::chain::ComputeChainObjfAndDeriv");

}
