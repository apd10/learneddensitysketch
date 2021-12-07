from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lma_embedding_bag',
    ext_modules=[CUDAExtension(
        'lma_embedding_bag', 
        ['lma_embedding_bag_kernel.cu'])],
    py_modules=['LMAEmbedding'],
    cmdclass={'build_ext': BuildExtension}
)
