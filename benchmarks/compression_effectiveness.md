# Compression Effectiveness Benchmarks

These benchmarks evaluate the neural compression capabilities of the Agent Memory System, focusing on how well it preserves information while reducing storage requirements.

## Embedding Quality

### Benchmark Description
Measure the information retention and quality of neural embeddings after compression across all memory tiers.

### Methodology
1. Generate diverse test datasets with varying semantic complexity
2. Process data through the autoencoder for each memory tier
3. Reconstruct original data from compressed embeddings
4. Measure reconstruction error and information retention

### Code Example
```python
from memory import AgentMemorySystem, MemoryConfig
from memory.embeddings.autoencoder import AutoEncoder
import numpy as np
from sklearn.metrics import mean_squared_error

def benchmark_embedding_quality(test_data_categories=["navigation", "dialogue", "inventory"]):
    results = {}
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    
    for category in test_data_categories:
        # Generate test data for this category
        test_data = generate_category_test_data(category, samples=100)
        
        # Test STM autoencoder (384-dim)
        stm_encoder = memory_system.get_encoder("stm")
        stm_embeddings = []
        reconstructions = []
        
        for item in test_data:
            # Convert to format expected by encoder
            prepared_item = prepare_for_encoding(item)
            
            # Encode and then decode
            embedding = stm_encoder.encode(prepared_item)
            stm_embeddings.append(embedding)
            
            # Reconstruct
            reconstruction = stm_encoder.decode(embedding)
            reconstructions.append(reconstruction)
        
        # Calculate metrics
        mse = mean_squared_error(
            np.array([prepare_for_encoding(x) for x in test_data]), 
            np.array(reconstructions)
        )
        cosine_similarity = calculate_average_cosine_similarity(
            np.array([prepare_for_encoding(x) for x in test_data]), 
            np.array(reconstructions)
        )
        
        results[category] = {
            "stm": {
                "reconstruction_mse": mse,
                "cosine_similarity": cosine_similarity
            },
            # Same calculations for IM and LTM
        }
    
    return results
```

### Expected Metrics
- Mean Squared Error (MSE) of reconstruction
- Cosine similarity between original and reconstructed data
- Domain-specific semantic retention metrics

## Compression Ratio

### Benchmark Description
Evaluate the size reduction achieved by each memory tier's compression techniques.

### Methodology
1. Generate test datasets of varying complexity
2. Store raw data in STM and compressed versions in IM and LTM
3. Measure storage size of each representation
4. Calculate compression ratios between tiers

### Code Example
```python
def benchmark_compression_ratio():
    memory_system = AgentMemorySystem.get_instance(MemoryConfig())
    test_dataset = generate_diverse_test_dataset(1000)
    
    # Calculate raw size
    raw_size = calculate_serialized_size(test_dataset)
    
    # STM embeddings (384-dim)
    stm_encoder = memory_system.get_encoder("stm")
    stm_embeddings = [stm_encoder.encode(prepare_for_encoding(item)) for item in test_dataset]
    stm_size = calculate_array_storage_size(stm_embeddings)
    
    # IM embeddings (128-dim)
    im_encoder = memory_system.get_encoder("im")
    im_embeddings = [im_encoder.encode(prepare_for_encoding(item)) for item in test_dataset]
    im_size = calculate_array_storage_size(im_embeddings)
    
    # LTM embeddings (32-dim)
    ltm_encoder = memory_system.get_encoder("ltm")
    ltm_embeddings = [ltm_encoder.encode(prepare_for_encoding(item)) for item in test_dataset]
    ltm_size = calculate_array_storage_size(ltm_embeddings)
    
    return {
        "raw_size_bytes": raw_size,
        "stm_size_bytes": stm_size,
        "im_size_bytes": im_size,
        "ltm_size_bytes": ltm_size,
        "stm_compression_ratio": raw_size / stm_size,
        "im_compression_ratio": raw_size / im_size,
        "ltm_compression_ratio": raw_size / ltm_size,
        "im_to_stm_ratio": stm_size / im_size,
        "ltm_to_im_ratio": im_size / ltm_size,
    }
```

### Expected Metrics
- Size in bytes for each representation
- Raw-to-compressed ratios
- Tier-to-tier compression ratios

## Training Efficiency

### Benchmark Description
Measure the computational resources required to train and update autoencoder models.

### Methodology
1. Time the initial training of autoencoder models for each tier
2. Measure incremental training time for new data types
3. Evaluate GPU/CPU resource utilization during training
4. Test inference speed on trained models

### Code Example
```python
import time
import psutil
import torch

def benchmark_autoencoder_training(data_sizes=[1000, 10000, 100000]):
    results = {}
    
    for size in data_sizes:
        training_data = generate_training_dataset(size)
        
        # STM autoencoder (384-dim)
        stm_config = {
            "input_dim": calculate_input_dimensions(training_data),
            "encoding_dim": 384,
            "hidden_layers": [512, 384]
        }
        stm_autoencoder = AutoEncoder(stm_config)
        
        # Measure training time and resource usage
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        if torch.cuda.is_available():
            start_gpu_memory = torch.cuda.memory_allocated()
        
        stm_autoencoder.train(training_data, epochs=10, batch_size=32)
        
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated()
            gpu_memory_used = end_gpu_memory - start_gpu_memory
        else:
            gpu_memory_used = 0
        
        # Measure inference speed
        inference_times = []
        test_samples = training_data[:100]  # Use subset for testing
        for sample in test_samples:
            inf_start = time.time()
            _ = stm_autoencoder.encode(sample)
            inference_times.append(time.time() - inf_start)
        
        results[size] = {
            "stm": {
                "training_time_seconds": end_time - start_time,
                "memory_used_bytes": end_memory - start_memory,
                "gpu_memory_used_bytes": gpu_memory_used,
                "avg_inference_time_ms": (sum(inference_times) / len(inference_times)) * 1000
            },
            # Similar measurements for IM and LTM autoencoders
        }
    
    return results
```

### Expected Metrics
- Initial training time (seconds)
- Memory usage during training (bytes)
- GPU utilization (if applicable)
- Average inference time (milliseconds)
- Convergence rate (loss vs. epochs) 