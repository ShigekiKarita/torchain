#pragma once

#include <type_traits>

#include <torch/extension.h>

// kaldi
#include "matrix/matrix-common.h"

namespace torchain
{

    static_assert(std::is_same<kaldi::BaseFloat, float>::value ||
                  std::is_same<kaldi::BaseFloat, double>::value,
                  "kaldi::BaseFloat should be float or double");

    constexpr at::ScalarType atBaseFloat = std::is_same<kaldi::BaseFloat, float>::value ? at::kFloat : at::kDouble;


    template <typename M>
    M make_submatrix_impl(torch::Tensor t)
    {
        static_assert(
            std::is_same<M, kaldi::SubMatrix<kaldi::BaseFloat>>::value ||
            std::is_same<M, kaldi::CuSubMatrix<kaldi::BaseFloat>>::value,
            "M must be kaldi::(Cu)SubMatrix<kaldi::BaseFloat>"
            );
        KALDI_ASSERT(t.dim() == 2);
        KALDI_ASSERT(t.scalar_type() == atBaseFloat);
        KALDI_ASSERT(t.device() ==
                     (std::is_same<M, kaldi::CuSubMatrix<kaldi::BaseFloat>>::value
                      ? at::kCUDA : at::kCPU));
        return M(t.data<kaldi::BaseFloat>(),
                 t.size(0),
                 t.size(1),
                 t.stride(0));
    }

    /// ref torch tensor to kaldi matrix
    inline kaldi::SubMatrix<kaldi::BaseFloat> make_submatrix(torch::Tensor t)
    {
        return make_submatrix_impl<kaldi::SubMatrix<kaldi::BaseFloat>>(t);
    }

    /// ref torch tensor to kaldi matrix
    inline kaldi::CuSubMatrix<kaldi::BaseFloat> make_cusubmatrix(torch::Tensor t)
    {
        return make_submatrix_impl<kaldi::CuSubMatrix<kaldi::BaseFloat>>(t);
    }

    /// copy kaldi matrix to torch tensor
    inline torch::Tensor copy_tensor(const kaldi::GeneralMatrix& mat)
    {
        auto opt = at::TensorOptions().device(at::kCPU).dtype(atBaseFloat);
        auto ret = torch::empty({mat.NumRows(), mat.NumCols()}, opt);
        auto retm = make_submatrix(ret);
        mat.CopyToMat(&retm);
        return ret;
    }

    /// copy kaldi matrix to torch tensor
    /// TODO(karita): support transposed
    inline torch::Tensor ref_tensor(kaldi::CuMatrixBase<kaldi::BaseFloat>& mat)
    {
        auto opt = at::TensorOptions().device(at::kCUDA).dtype(atBaseFloat);
        auto ref = at::from_blob(
            mat.Data(),
            {mat.NumRows(), mat.NumCols()},
            {mat.Stride(), 1},
            [](void*) {}, // not to free
            opt
            );
        return ref;
    }

} // namespace torchain
