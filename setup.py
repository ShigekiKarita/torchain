import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


KALDI_ROOT = os.path.realpath(os.environ.get("KALDI_ROOT", "../../../"))
assert os.path.exists(KALDI_ROOT + "/src/lib/libkaldi-chain.so"), "need to set $KALDI_ROOT"

# make cu-device-patched.h
patch_dir = "patched"
os.makedirs(patch_dir, exist_ok=True)
with open(patch_dir + "/cu-device-patched.h", "w") as wf:
    with open(KALDI_ROOT + "/src/cudamatrix/cu-device.h", "r") as rf:
        is_cu_device = False
        n_rewrite = 0
        for line in rf:
            if "class CuDevice" in line:
                is_cu_device = True
            if "}; // class CuDevice" in line:
                is_cu_device = False
            if is_cu_device and "private:" in line:
                line = line.replace("private:", "public:")
                n_rewrite += 1
            wf.write(line)
        assert not is_cu_device
        assert n_rewrite == 1


extra_compile_args = [
    "-DHAVE_CUDA=1", "-DKALDI_PARANOID",
    "-isystem" + patch_dir,
    "-isystem" + KALDI_ROOT + "/src",
    "-isystem" + KALDI_ROOT + "/tools/openfst/include",
    "-std=c++14"
]


extra_link_args = [
    "-L" + KALDI_ROOT + "/src/lib",
    "-Wl,-rpath=" + KALDI_ROOT + "/src/lib",
    "-lkaldi-chain", "-lkaldi-nnet3", "-lkaldi-util"
]


setup(
    name="torchain",
    packages=["torchain"],
    package_dir={"": "python"},
    # build setup
    ext_modules=[
        CppExtension(
            name="torchain.egs",
            sources=["cxx/egs.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        ),
        # TODO(karita): CUDAExtention?
        CppExtension(
            name="torchain.train",
            sources=["cxx/train.cpp"],
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    # test setup
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
