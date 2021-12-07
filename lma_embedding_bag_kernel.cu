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
#define MAX_BLOCK_SIZE  128


__device__ int64_t hash_func(int64_t a, int64_t b, const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers) {
    return (a * random_numbers[3] + b * random_numbers[2] + random_numbers[1]) % random_numbers[0];
}

inline __device__ int location2d(int i,int j, int I, int J) {
    return i * J + j; // rowmajor
}

template<typename scalar_t>
__global__ void lma_embedding_bag_update_output_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 3, torch::RestrictPtrTraits> random_projections,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> hashed_weights,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> output,
    torch::PackedTensorAccessor32<int64_t, 3, torch::RestrictPtrTraits> hashed_index,
    const torch::PackedTensorAccessor32<int64_t, 1, torch::RestrictPtrTraits> random_numbers,
    int batch_size,
    int array_size,
    int embedding_dim,
    int num_rep,
    int num_chunks,
    int bits_per_chunk,
    int chunk_size,
    int64_t hashedWeightSize,
    int lsh_mode_enum)
{
    
    int x_idx = blockIdx.x; // for sample id  - need looping
    int y_idx = threadIdx.y; // for chunk id - need looping
    int z_idx = threadIdx.z; // repetition number - no looping
    
    ////printf("x_idx: %d, y_idx: %d, z_idx:%d  blockdimensions:(%d,%d,%d) griddim:(%d,%d,%d)\n", 
    //        x_idx, y_idx, z_idx, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
    if ( x_idx < batch_size and y_idx < num_chunks and z_idx < num_rep) {
      int64_t location = 0;
      int64_t power2 = 1;
      int bit_offset = 0;
      int chunk_offset = 0; 
      int local_chunk_offset = 0;
      int local_max_total_chunk = chunk_size * blockDim.y;
      int64_t idx;
      int64_t final_idx;
      int stride;
      int num;
      // for per block. blockDim.x=1 * blockDim.y * blockDim.z * chunk_size
      extern __shared__ float local_output[];
      bool even;
      for(int i=x_idx; i < batch_size; i+= gridDim.x) {
        for(int j = y_idx; j < num_chunks; j+= blockDim.y) {
          //printf("example: %d(+%d) chunk_id:%d(+%d) repetition:%d\n", i, gridDim.x, j, blockDim.y, z_idx);

          // i is the example id
          // j is the chunk id
          location = 0;
          power2 = 1;
          bit_offset = bits_per_chunk * j;
          for(int k = 0; k < bits_per_chunk ; k++) {
            //location += ((random_projections[i][z_idx][bit_offset + k] > 0) ? power2 : 0);
            if (random_projections[i][z_idx][bit_offset + k] > 0) {
              location = location + power2;
              //printf("[%d,%d,%d] power:%ld location:%ld\n", i,j,z_idx, power2,location);
            }
            //printf("[%d,%d,%d] loop : using projection: (%d,%d,%d) value: %f bit:%d inchunk:%d power2:%ld location:%ld\n", i,j,z_idx, i, z_idx, bit_offset + k, random_projections[i][z_idx][bit_offset + k], (random_projections[i][z_idx][bit_offset + k] > 0), k, power2, location);

            power2 = power2<<1L;
          }
          idx = hash_func(location, j, random_numbers) % array_size;
          //printf("[%d,%d,%d] location: %ld idx: %ld\n", i,j,z_idx, location, idx);
          chunk_offset = chunk_size * j;
          local_chunk_offset = chunk_size * (j % blockDim.y); // only need to look at the block

          ////printf("x_idx:%d z_idx:%d i:%d j:%d loc:%ld idx:%ld \n", x_idx,z_idx,i,j, location, idx);
          for(int k=0; k < chunk_size ; k++) {
            final_idx = z_idx * array_size + (idx + k) % array_size;
            hashed_index[i][z_idx][chunk_offset + k] = final_idx;
            local_output[location2d(z_idx,local_chunk_offset+ k,num_rep, local_max_total_chunk)] = hashed_weights[final_idx];
            //printf("[%d,%d,%d] idx: %ld final_idx: %ld hashed_weight: %f \n", i,j,z_idx, idx, final_idx, hashed_weights[final_idx]);
          }
          // need to reduce local_output into output 
          num = num_rep;
          while (num > 1) {
            even = ((num & 1) == 0);
            stride =  num / 2;
            if( z_idx < stride) {
              for(int k=0; k < chunk_size ; k++) {
                local_output[location2d(z_idx, local_chunk_offset + k, num_rep, local_max_total_chunk)] += local_output[location2d(z_idx+stride, local_chunk_offset + k, num_rep, local_max_total_chunk)];
              }
            }
            num = stride + (even ? 0 : 1);
            if(i == 0 and j==0 and z_idx == 0) {
              //printf("i,j:(%d,%d) : num: %d stride:%d even:%d\n", i, j, num, stride, even);
            }
            __syncthreads(); // sync block
          }
          if(z_idx == 0) {
            for(int k=0; k < chunk_size ; k++) {
              output[i][chunk_offset + k] = local_output[location2d(z_idx, local_chunk_offset + k, num_rep, local_max_total_chunk)] / num_rep;
              //printf("[%d,%d,%d] ouput: %f\n",  i,j,z_idx, output[i][chunk_offset + k]);
            }
          }
        }
      }
    }
}


std::tuple<torch::Tensor, torch::Tensor> lma_embedding_bag_cuda_forward(
    const torch::Tensor& hashed_weights, // 1 x n
    const torch::Tensor& input_embeddings, // b x d1
    const torch::Tensor& lsh_matrix, // d1 x num_rep x chunks
    const torch::Tensor& random_numbers,
    int input_dim,
    int embedding_dim,
    int num_rep,
    int num_chunks,
    int chunk_size,
    int bits_per_chunk,
    int lsh_mode_enum
    )
{

  auto lsh_matrix_flat = at::reshape(lsh_matrix, {input_embeddings.size(1), -1});
  auto random_projections = at::matmul(input_embeddings, lsh_matrix_flat); // b x num_rep x (num_chunks*bits_per_chunk)
  random_projections = at::reshape(random_projections, {input_embeddings.size(0), num_rep, -1});
    cudaDeviceSynchronize();

    int64_t hashedWeightSize = hashed_weights.size(0);
    auto hashed_index = at::empty({input_embeddings.size(0), num_rep, embedding_dim}, input_embeddings.options().dtype(torch::kInt64)); // everything is same but it is int64 and not float
    auto output = at::empty({input_embeddings.size(0), embedding_dim}, input_embeddings.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream(input_embeddings.device().index()); 

#ifdef __HIP_PLATFORM_HCC__
    dim3 block = dim3(64, 4);
#else
    dim3 block = dim3(32, 8);
#endif
    int grid = 1024;
    if( num_rep > 64) {
      //printf("Too many repetitions not supported, max: %d\n", MAX_BLOCK_SIZE);
      assert(false);
    }
    int max_chunks = (int)((MAX_BLOCK_SIZE + num_rep - 1) /num_rep);
    if (num_chunks <= max_chunks) {
        block = dim3(1, num_chunks, num_rep);
    } else {
        block = dim3(1, max_chunks, num_rep);
    }
    
    if (input_embeddings.size(0) < MAX_GRID_SIZE) {
        grid = input_embeddings.size(0);
    } else {
        grid = MAX_GRID_SIZE;
    }

    //printf("Calling the kernel grid:%d block:(1,%d,%d) \n", grid, num_chunks, num_rep); fflush(stdout);
    AT_DISPATCH_FLOATING_TYPES(hashed_weights.type(), "lma_embedding_bag_cuda", ([&] {
        lma_embedding_bag_update_output_kernel<scalar_t><<<grid, block, (num_rep * block.y * chunk_size * sizeof(float)), stream>>>(
            random_projections.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>(),
            hashed_weights.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            hashed_index.packed_accessor32<int64_t, 3, torch::RestrictPtrTraits>(),
            random_numbers.packed_accessor32<int64_t, 1, torch::RestrictPtrTraits>(),
            random_projections.size(0),
            (int) hashed_weights.size(0) / num_rep,
            embedding_dim,
            num_rep,
            num_chunks,
            bits_per_chunk,
            chunk_size,
            hashedWeightSize,
            lsh_mode_enum);
    }));
    fflush(stdout);
    cudaDeviceSynchronize();  
   return std::tuple<torch::Tensor, torch::Tensor>(
        output, hashed_index);
}

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::tuple<torch::Tensor, torch::Tensor> lma_embedding_bag_forward(
    const torch::Tensor& hashed_weights,
    const torch::Tensor& input_embeddings,
    const torch::Tensor& lsh_matrix,
    const torch::Tensor& random_numbers,
    int input_dim,
    int embedding_dim,
    int num_rep,
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
                                          input_dim, embedding_dim, num_rep, num_chunks, chunk_size, 
                                          bits_per_chunk, lsh_mode_enum);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lma_embedding_bag_forward, "lma embedding forward (CUDA)");
}
