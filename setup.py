import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension


KALDI_ROOT = os.path.realpath(os.environ.get("KALDI_ROOT", "../../../"))
assert os.path.exists(KALDI_ROOT + "/src/lib/libkaldi-chain.so"), "need to set $KALDI_ROOT"


setup(
    name="torchain",
    packages=["torchain"],
    package_dir={"": "python"},
    # build setup
    ext_modules=[
        CppExtension(
            name="torchain.egs",
            sources=["cxx/egs.cpp"],
            extra_compile_args=[
                "-DHAVE_CUDA=1",
                "-isystem" + KALDI_ROOT + "/src",
                "-isystem" + KALDI_ROOT + "/tools/openfst/include",
            ],
            extra_link_args=[
                "-L" + KALDI_ROOT + "/src/lib",
                "-Wl,-rpath=" + KALDI_ROOT + "/src/lib",
                "-lkaldi-chain", "-lkaldi-nnet3", "-lkaldi-util"
            ]
        ),
        # TODO(karita): CUDAExtention?
        CppExtension(
            name="torchain.train",
            sources=["cxx/train.cpp"],
            extra_compile_args=[
                "-DHAVE_CUDA=1",
                "-isystem" + KALDI_ROOT + "/src",
                "-isystem" + KALDI_ROOT + "/tools/openfst/include",
            ],
            extra_link_args=[
                "-L" + KALDI_ROOT + "/src/lib",
                "-Wl,-rpath=" + KALDI_ROOT + "/src/lib",
                "-lkaldi-chain", "-lkaldi-nnet3", "-lkaldi-util"
            ]
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    # test setup
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
