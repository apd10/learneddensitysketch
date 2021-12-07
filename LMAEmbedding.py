from typing import Optional

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from torch.nn.parameter import Parameter
import math
import lma_embedding_bag
import pdb

def weighted_values(values, probabilities, size):
    bins = np.add.accumulate(probabilities)
    return values[np.digitize(np.random.random_sample(size), bins)]

def generateSparseSRP(N, d):
    _v = np.array([1, -1, 0])
    _prob = np.array([0.1667,0.1667, 0.6666])
    return weighted_values(_v, _prob, (d*N)).reshape(N, d)


class LMAEmbeddingFunction(torch.autograd.Function):
    @staticmethod

    def forward(ctx, hashed_weights, input_embeddings, lsh_matrix, random_numbers, input_dim, embedding_dim,
                    num_rep, num_chunks, chunk_size, bits_per_chunk, lsh_mode):
        '''
            read a chunk_size by performing lsh according to the lsh_mode,
            join chunks to create an embedding of size embedding_dim for each of the
            inputs
        '''
        if lsh_mode == 'srp':
            lsh_mode_enum = 0
        else:
            raise ValueError("lsh_mode not defined:", lsh_mode)
        
        
        output, hashed_idx = \
            lma_embedding_bag.forward(hashed_weights, input_embeddings, lsh_matrix, random_numbers,  input_dim, embedding_dim, num_rep, num_chunks, chunk_size, bits_per_chunk, lsh_mode_enum)

        
        ctx.save_for_backward(hashed_idx)
        ctx.hashed_weights_size = hashed_weights.shape[0]
        return hashed_idx, output

    '''
    @staticmethod
    def backward(ctx, grad):
        hashed_idx, input_dim, embedding_dim, num_chunks, chunk_size, bits_per_chunk, lsh_mode = ctx.saved_variables
        weight_grad = hashed_embedding_bag.backward(
                grad, hashed_idx, input_dim, embedding_dim, num_chunks, chunk_size, bits_per_chunk, lsh_mode)
        return weight_grad, None, None, None, None, None, None
    '''

    @staticmethod
    def backward(ctx, grad):
        hashed_idx, = ctx.saved_variables
        hashed_weights_size = ctx.hashed_weights_size

        weight_grad = torch.zeros(hashed_weights_size).to(grad.device)

        if grad.is_contiguous():
            grad1 = grad.view(-1)
        else:
            grad1 = grad.reshape(-1)
      
        for i in range(hashed_idx.shape[1]):
            hashed_idx1 = hashed_idx[:,i,:].reshape(-1)
            weight_grad.scatter_add_(0, hashed_idx1, grad1)
        weight_grad = weight_grad / hashed_idx.shape[1]

        return weight_grad, None, None, None, None, None, None, None, None, None, None

class LMAEmbedding(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        embedding_dim: int, 
        chunk_size: int,
        bits_per_chunk: int,
        num_rep: int,
        hashed_weight: torch.Tensor,
        lsh_mode = "srp",
        seed = 1024)->None:
        super(LMAEmbedding, self).__init__()


        self.embedding_dim = embedding_dim
        self.lsh_mode = lsh_mode
        self.chunk_size = chunk_size
        self.bits_per_chunk = bits_per_chunk
        self.input_dim = input_dim
        self.num_rep = num_rep
        self.weight = hashed_weight
        self.memory_size = hashed_weight.size(0)
        self.array_size = self.memory_size / num_rep
        self.num_chunks = int(embedding_dim / chunk_size)

        assert(embedding_dim % chunk_size == 0)
        assert(self.memory_size % num_rep == 0)


        print("[single]Potential Memory Usage: ", 2**self.bits_per_chunk * self.num_chunks,  
              "memory available", self.array_size)
        print("[total ]Potential Memory Usage: ", 2**self.bits_per_chunk * self.num_chunks * self.num_rep,  
              "memory available", self.memory_size)

        if (lsh_mode == "srp"):
            self.lsh_matrix = Parameter(torch.from_numpy(generateSparseSRP(input_dim, self.num_chunks * self.bits_per_chunk * self.num_rep).astype(np.float32).reshape(input_dim, self.num_rep, self.num_chunks * self.bits_per_chunk)), requires_grad=False)
        else:
            raise ValueError("No lsh mode found")

        r = np.random.RandomState(seed)
        # first number is the prime, rest are random integers
        random_numbers = np.concatenate([np.array([2038074743]), r.randint(0, 2038074743, (50,))]) # set of 50 random numbers to use
        self.random_numbers = Parameter(torch.from_numpy(random_numbers.astype(np.int64)), requires_grad=False)
        print("RandomNumbers: ", self.random_numbers[:5])
        
        print("LMAEmbedding: input_dim: ", input_dim, " embedding_dim:", embedding_dim, "rep:", self.num_rep, "chunk_size:",
              chunk_size, " bits_per_chunk:", bits_per_chunk, "lsh_mode:", lsh_mode, " seed:", seed, 
              " lsh matrix", self.lsh_matrix.shape,  "memory_size:", self.memory_size, "array_size:", self.array_size)

    def forward(self, input_embeddings) -> torch.Tensor:
        #i_shape = indices.shape
        #indices = indices.view(-1)
        #if offsets is None:
        #    offsets  = torch.arange(len(indices)).to(indices.device)
        #assert(per_sample_weights is None)
        embeddings =  LMAEmbeddingFunction.apply(
            self.weight, input_embeddings, self.lsh_matrix, self.random_numbers,  self.input_dim, self.embedding_dim, self.num_rep, self.num_chunks, self.chunk_size, self.bits_per_chunk, self.lsh_mode
        )
        return embeddings

