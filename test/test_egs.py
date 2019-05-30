import os
import torchain


cwd = os.path.dirname(os.path.realpath(__file__))


if __name__ == "__main__":
    feat = cwd + "/res/mfcc.ark"
    supv = cwd + "/res/supervision.ark"
    get_egs = torchain.GetEgs(feat="ark,s,cs:" + feat, supervision="ark:" + supv)
    egs = get_egs.load("1028-20100710-hne-ar-01")
    assert len(egs) > 0
    # flatten chunks
    for i, c in enumerate(get_egs.chunks()):
        print(c)
    assert i == 41

    for e in egs:
        print(e.key, [(k, v) for k, v in e.outputs.items()])
        print(e.key, [(k, v.shape) for k, v in e.inputs.items()])
