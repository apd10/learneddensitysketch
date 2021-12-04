#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <ATen/core/TensorAccessor.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/unique.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/host_vector.h>

#include <vector>
#include <stdio.h>

#define MAX_GRID_SIZE  2048
#define MAX_BLOCK_SIZE  512


__device__ int64_t hash_func(int64_t a, int64_t b, const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers) {
    return (a * random_numbers[3] + b * random_numbers[2] + random_numbers[1]) % random_numbers[0];
}

template<typename scalar_t>
__global__ void lma_embedding_bag_update_output_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> random_projections,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> hashed_weights,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<int64_t, 2, torch::RestrictPtrTraits> hashed_index,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers,
    int batch_size,
    int embedding_dim,
    int num_chunks,
    int bits_per_chunk,
    int chunk_size,
    int64_t hashedWeightSize,
    int lsh_mode_enum)
{
    
    int x_idx = blockIdx.x;
    int y_idx = threadIdx.y;
    if(x_idx == 0 and y_idx ==0) {
      printf("bits_per_chunk: %d\n", bits_per_chunk);
    }   
    
    int64_t location = 0;
    int64_t power2 = 1;
    int bit_offset = 0;
    int chunk_offset = 0; 
    int64_t idx;
    int64_t final_idx;
    for(int i=x_idx; i < batch_size; i+= gridDim.x) {
        for(int j = y_idx; j < num_chunks; j+= blockDim.y) {
          // i is the example id
          // j is the chunk id
          location = 0;
          power2 = 1;
          bit_offset = bits_per_chunk * j;
          for(int k = 0; k < bits_per_chunk ; k++) {
              location += ((random_projections[i][bit_offset + k] > 0) ? power2 : 0);
              power2 = power2<<1;
              printf("RP: %d %d %f \n",i, bit_offset + k, random_projections[i][bit_offset + k]);
          }

          idx = hash_func(location, j, random_numbers) % hashedWeightSize;
          chunk_offset = chunk_size * j;

          printf("x_idx:%d y_idx:%d i:%d j:%d loc:%ld idx:%ld \n", x_idx,y_idx,i,j, location, idx);
          for(int k=0; k < chunk_size ; k++) {
              final_idx = (idx + k) % hashedWeightSize;
              hashed_index[i][chunk_offset + k] = final_idx;
              output[i][chunk_offset + k] = hashed_weights[final_idx];
          }
        }
    }
}


std::tuple<torch::Tensor, torch::Tensor> lma_embedding_bag_cuda_forward(
    const torch::Tensor& hashed_weights, // 1 x n
    const torch::Tensor& input_embeddings, // b x d1
    const torch::Tensor& lsh_matrix, // d1 x num_chunks
    const torch::Tensor& random_numbers,
    int input_dim,
    int embedding_dim,
    int num_chunks,
    int chunk_size,
    int bits_per_chunk,
    int lsh_mode_enum
)
{

    auto random_projections = at::mm(input_embeddings, lsh_matrix); // b x (num_chunks*bits_per_chunk)
    cudaDeviceSynchronize();

    int64_t hashedWeightSize = hashed_weights.size(0);
    auto hashed_index = at::empty({input_embeddings.size(0), embedding_dim}, input_embeddings.options().dtype(torch::kInt64)); // everything is same but it is int64 and not float
    auto output = at::empty({input_embeddings.size(0), embedding_dim}, input_embeddings.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input_embeddings.device().index()); 

#ifdef __HIP_PLATFORM_HCC__
    dim3 block = dim3(64, 4);
#else
    dim3 block = dim3(32, 8);
#endif
    int grid = 1024;

    if (num_chunks <= MAX_BLOCK_SIZE) {
        block = dim3(1,num_chunks);
    } else {
        block = dim3(1, MAX_BLOCK_SIZE);
    }
    
    if (input_embeddings.size(0) < MAX_GRID_SIZE) {
        grid = input_embeddings.size(0);
    } else {
        grid = MAX_GRID_SIZE;
    }
    printf("Calling the kernel\n"); fflush(stdout);
    AT_DISPATCH_FLOATING_TYPES(hashed_weights.type(), "lma_embedding_bag_cuda", ([&] {
        lma_embedding_bag_update_output_kernel<scalar_t><<<grid, block, 0, stream>>>(
            random_projections.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            hashed_weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            hashed_index.packed_accessor32<int64_t, 2, torch::RestrictPtrTraits>(),
            random_numbers.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            random_projections.size(0),
            embedding_dim,
            num_chunks,
            bits_per_chunk,
            chunk_size,
            hashedWeightSize,
            lsh_mode_enum);
    }));
    cudaDeviceSynchronize();  
   return std::tuple<torch::Tensor, torch::Tensor>(
        output, hashed_index);
}

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



//lma_embedding_bag.forward(hashed_weights, input_embeddings, lsh_matrix, input_dim, embedding_dim, num_chunks, chunk_size, bits_per_chunk, lsh_mode)
        
std::tuple<torch::Tensor, torch::Tensor> lma_embedding_bag_forward(
    const torch::Tensor& hashed_weights,
    const torch::Tensor& input_embeddings,
    const torch::Tensor& lsh_matrix,
    const torch::Tensor& random_numbers,
    int input_dim,
    int embedding_dim,
    int num_chunks,
    int chunk_size,
    int bits_per_chunk,
    int lsh_mode_enum
)
{
  
    CHECK_INPUT(hashed_weights);
    CHECK_INPUT(input_embeddings);
    CHECK_INPUT(lsh_matrix);
    CHECK_INPUT(random_numbers);
    

    return lma_embedding_bag_cuda_forward( hashed_weights, input_embeddings, lsh_matrix, random_numbers,
                                          input_dim, embedding_dim, num_chunks, chunk_size, bits_per_chunk, lsh_mode_enum);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lma_embedding_bag_forward, "lma embedding forward (CUDA)");
}
