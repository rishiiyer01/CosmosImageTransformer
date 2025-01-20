# CosmosImageTransformer

The purpose of this repo was to benchmark my experimental latent image architectures with an extremely small, lightweight causal transformer applied to the sequence of latent patches from cosmos tokenizer in raster scan order. Model is ~70M params.








Use distributedtrain.py for multi-gpu training (fsdp but no sharding, just data parallel), use train.py for single gpu training. You will likely need to adapt code to shard without H100 or A100 gpus.
