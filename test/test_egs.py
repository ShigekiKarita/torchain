import os
import torchain


cwd = os.path.dirname(os.path.realpath(__file__))

def test_ark():
    feat = cwd + "/mfcc1.ark"
    supv = cwd + "/supervision1.ark"
    if not os.path.exists(feat):
        print("WARNING: download ark files into test/ from https://github.com/ShigekiKarita/voxforge-chain/releases/tag/pybind-test")
        return
    get_egs = torchain.GetEgs(feat=feat, supv=supv)
    egs = torchain.Example()
    get_egs.load(0, egs)
