import os
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension

# TODO(karita) automate conf
KALDI_ROOT = os.path.realpath(os.environ.get("KALDI_ROOT", "../../../"))

setup(
    name="torchain",
    packages=["torchain"],
    package_dir={"": "python"},
    ext_modules=[
        CppExtension(
            name="torchain.egs",
            sources=["cxx/egs.cpp"],
            extra_compile_args=[
                "-isystem" + KALDI_ROOT + "/src",
                "-isystem" + KALDI_ROOT + "/tools/openfst/include",
            ],
            extra_link_args=["-L" + KALDI_ROOT + "/src/lib",
                             "-Wl,-rpath=" + KALDI_ROOT + "/src/lib",
                             "-lkaldi-chain", "-lkaldi-nnet3"]
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
