nvbio
=====

NVBIO is a library of reusable components designed by NVIDIA Corporation to
accelerate bioinformatics applications using CUDA.

Though it is specifically designed to unleash the power of NVIDIA GPUs, most of
its components are completely cross-platform and can be used both from host C++
and device CUDA code.

The purpose of NVBIO is twofold: it can be thought of both as a solid basis to
build new, modern applications targeting GPUs, which deferring the core
computations to a library will always automatically and transparently benefit
from new advances in GPU computing, and as example material to design novel
bioinformatics algorithms for massively parallel architectures.

Additionally, NVBIO contains a suite of applications built on top of it,
including a re-engineered implementation of the famous Bowtie2 short read
aligner. Unlike many prototypes, nvBowtie is an attempt to build an industrial
strength aligner, reproducing most of Bowtie2's original features as well as
adding a few more, such as efficient support for direct BAM (and soon CRAM)
output.

NVBIO is hosted on GitHub at `http://nvlabs.github.io/nvbio/`.



Compilation
-----------

To compile, you first need to acquire submodules.

You can do this while cloning using:

    git clone --recursive git@github.com:vmiheer/nvbio.git

You can do this after having cloned using:

    git submodule update --init --recursive

After this, you must build. To do so, enter the nvbio base directory and perform the following:

    mkdir build
    cd build
    cmake ..
    make -j8

For CUDA 9, you must use GCC6 or less:

    CXX=g++-6 CC=gcc-6 cmake ..

For CUDA 10, you can use up to GCC8.2.

CMake options:

 * `-DGPU_ARCHITECTURE=sm_XX` - By default NVBIO will use sm_35.
 * `-DCMAKE_BUILD_TYPE=Debug` - Compiles with debugging flags. By default full optimizations are used.


Testing
-------
Running ./nvbio-test/nvbio-test will give the following error: 
 ` warning : unable to open bwt "./data/human.NCBI36/Human.NCBI36.bwt" 
   error   :     failed opening file "./data/SRR493095_1.fastq.gz" `

You can obtain the file here https://www.ncbi.nlm.nih.gov/sra/SRX145461

Credits
-------

The main contributors of NVBIO are:

 * Jacopo Pantaleoni  -  jpantaleoni@nvidia.com
 * Nuno Subtil        -  nsubtil@nvidia.com
