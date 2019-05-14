// third party
#include <torch/extension.h>

// kaldi
#include "chain/chain-training.h"

#include "./tensor.hpp"

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

struct ChainResults
{
    BaseFloat objf;
    BaseFloat l2_term;
    BaseFloat weight;
};

// TODO(karita): use libtorch API to create new method?
void chain_loss(
    // inputs
    const kaldi::chain::DenominatorGraph& den_graph,
    const kaldi::chain::Supervision& supervision,
    torch::Tensor nnet_output_tensor, // CUDA
    // outputs
    ChainResults& results,
    // grads CUDA
    torch::Tensor nnet_output_deriv_tensor,
    torch::Tensor xent_output_deriv_tensor,
    // hyper params
    const kaldi::chain::ChainTrainingOptions& opts)
{
    // TODO(karita):
    // set_kaldi_device(nnet_output_ptr);
    auto nnet_output = torchain::make_cusubmatrix(nnet_output_tensor);
    auto nnet_output_deriv = torchain::make_cusubmatrix(nnet_output_deriv_tensor);

    kaldi::CuMatrix<BaseFloat> xent_deriv;
    auto xent_deriv_ptr = opts.xent_regularize != 0.0 ? &xent_deriv : nullptr;
    kaldi::chain::ComputeChainObjfAndDeriv(opts, den_graph, supervision, nnet_output,
                                           &results.objf, &results.l2_term, &results.weight,
                                           &nnet_output_deriv, xent_deriv_ptr);
    if (opts.xent_regularize != 0.0)
    {
        xent_output_deriv_tensor.copy_(torchain::ref_tensor(xent_deriv));
    }
}


// NOTE: these pybind11 symbols are imported from <torch/extension.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

    m.def("chain_loss", &chain_loss, "wrapper of kaldi::chain::ComputeChainObjfAndDeriv");

}
