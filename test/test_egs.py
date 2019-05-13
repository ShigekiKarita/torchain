import os
import torchain


cwd = os.path.dirname(os.path.realpath(__file__))

def test_ark():
    feat = cwd + "/mfcc1.ark"
    supv = cwd + "/supervision1.ark"
    for path in (feat, supv):
        assert os.path.exists(path), "ERROR: download ark files into test/ from https://github.com/ShigekiKarita/voxforge-chain/releases/tag/pybind-test"

    get_egs = torchain.GetEgs(feat="ark,s,cs:" + feat, supervision="ark:" + supv)
    # egs = [] # torchain.Example()
    egs = get_egs.load("1028-20100710-hne-ar-02")
    print(egs)
    assert len(egs) > 0


if __name__ == "__main__":
    test_ark()
