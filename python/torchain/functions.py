import torch

from torchain.train import chain_loss, ChainTrainingOptions


class _ChainLoss(torch.autograd.Function):
    def __init__(self, den_graph, l2_regularize=0.0,
                 leaky_hmm_coefficient=1e-5, xent_regularize=0.0):
        super().__init__()
        self.den_graph = den_graph
        self.opt = ChainTrainingOptions()
        self.opt.l2_regularize = l2_regularize
        self.opt.leaky_hmm_coefficient = leaky_hmm_coefficient
        self.opt.xent_regularize = xent_regularize
        self.output_deriv = torch.tensor()
        self.xent_deriv = torch.tensor()

    def forward(self, nnet_output, supervision):
        self.output_deriv = torch.empty_like(nnet_output)
        self.result = chain_loss(self.den_graph, supervision, nnet_output,
                            self.output_deriv, self.xent_deriv, self.opt)
        # TODO(karita) what?
        # https://github.com/kaldi-asr/kaldi/blob/182f3829e1afdb7fe94eafe24ea066b328d2cd9f/src/nnet3/nnet-chain-training.cc#L320
        return nnet_output.new([-self.result.objf / self.result.weight])

    def backward(self, grad_output):
        return self.output_deriv, None


def to2d(x):
    if x.dim() == 3:  # (B, C, T)
        # TODO double-check this
        n_pdf = x.shape[1]
        # x = x.transpose(1, 2).contiguous().view(-1, n_pdf)  # (B * T, C)
        x = x.permute(2, 0, 1).contiguous().view(-1, n_pdf)  # (T * B, C)
    assert(x.dim() == 2)
    return x


def chain_loss(nnet_output, supervision, den_graph, xent_output=None,
               l2_regularize=0.0, leaky_hmm_coefficient=1e-5, xent_regularize=0.0):
    func = _ChainLoss(den_graph, l2_regularize, leaky_hmm_coefficient, xent_regularize)
    nnet_output = to2d(nnet_output)
    loss = func(nnet_output, supervision)
    if xent_regularize != 0.0 and xent_output is not None:
        xent_output = to2d(xent_output)
        xent = xent_output.matmul(func.xent_deriv.transposed()).trace()
        func.result.xent = float(xent)
        loss = loss - xent_regularize * xent
    return loss, func.result
