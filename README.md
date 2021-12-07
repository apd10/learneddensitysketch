LMA Embedding v1.0
    
        - currently supports srp (sparse)
        - the parameters it takes is
            input_dim
            embedding_dim
            chunk_size : for chunk size support as in ROBE-Z. This controls the latency of access
                and hence the latency of inference
            num_bits_per_chunk : this will control the probability of collision
            num_rep : while building density sketch / embedding, we can use multiple repetitions
                for a single repetition, all elements of output embedding are hashed from a single
                array ( using universal  hasing). The reason for this choice is to not have dependence
                on num_bits. so you can use a lot of numbits / go to really low memory usage
                however, size of weight array should be divisible by num_rep as it hashes every rep
                in separate array space



```

# parameters to the LMAEmbedding
lsh_size = 10
input_dim = 10
embedding_dim = 10
bits_per_chunk = 4 # controls the probability of collison
chunk_size = 2
num_chunks = embedding_dim / chunk_size
num_rep = 1

# initialize weight parameter 
_weight = nn.Parameter(torch.from_numpy(-1 + 2 * np.random.binomial(1, 0.5, size=((lsh_size,))).astype(np.float32)))
emb = LMAEmbedding(input_dim, embedding_dim, chunk_size, bits_per_chunk, num_rep, _weight, seed=1).to(0)

torch.manual_seed(1)
inp = torch.rand(5,input_dim).to("cuda:0")
outp = emb(inp)
```
