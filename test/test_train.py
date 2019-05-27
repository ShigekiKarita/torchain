import os

import torch
import torchain


cwd = os.path.dirname(os.path.realpath(__file__))

def test_train():
    feat = cwd + "/res/mfcc.ark"
    supv = cwd + "/res/supervision.ark"
    get_egs = torchain.GetEgs(feat="ark,s,cs:" + feat, supervision="ark:" + supv)
    egs = get_egs.load("1028-20100710-hne-ar-01")
    y = egs[0].outputs["output"]
    x = egs[0].inputs["input"]
    den_fst = torchain.denominator_graph(cwd + "/res/den.fst", y.label_dim)
    model = torch.nn.Linear(x.shape[1], y.label_dim)

    if not torch.cuda.is_available():
        print("WARNING: cuda is not available. skip testing.")
    model.cuda()
    pred = model(x.cuda())
    print(pred.shape, y)
    loss = torchain.chain_loss(pred, y, den_fst)
    loss.backward()
    print(model.bias.grad)

if __name__ == "__main__":
    test_train()
