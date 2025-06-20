# Docker Configuration Options

This project supports different Docker configurations for various hardware setups.

## Basic Configuration (Default)
```bash
docker-compose up
```
- Uses 4GB shared memory
- Limited CPU workers
- No GPU support

## GPU Configuration (NVIDIA)
```bash
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

**Requirements:**
- NVIDIA GPU with CUDA support
- nvidia-docker2 installed
- nvidia-container-runtime

**Installation on Ubuntu:**
```bash
# Install NVIDIA Container Runtime
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
sudo systemctl restart docker
```

**Features:**
- Full GPU access for PyTorch/CUDA
- 16GB memory limit (configurable)
- 8GB shared memory
- Optimized for GPU training

## High-Memory CPU Configuration
```bash
docker-compose -f docker-compose.yml -f docker-compose.highmem.yml up
```

**Features:**
- 32GB memory limit (configurable)
- 8 CPU cores limit
- 16GB shared memory
- Optimized for large CPU training

## Memory Configuration Guidelines

### Adjust Memory Limits
Edit the configuration files and change:

```yaml
deploy:
  resources:
    limits:
      memory: 32G  # Change this value
    reservations:
      memory: 8G   # And this value
```

### Shared Memory Guidelines
- **Basic training**: 4GB
- **GPU training**: 8GB
- **Large datasets**: 16GB+
- **Multiple workers**: 2GB per worker

### Recommended Settings by System

| System RAM | Configuration | Memory Limit | Shared Memory |
|------------|--------------|--------------|---------------|
| 8GB        | Basic        | 6G           | 2gb           |
| 16GB       | Basic/GPU    | 12G          | 4gb           |
| 32GB       | High-mem     | 24G          | 8gb           |
| 64GB+      | High-mem     | 48G          | 16gb          |

## Troubleshooting

### "Out of shared memory" errors
- Increase `shm_size` in docker-compose files
- Reduce `num_workers` in training forms
- Use smaller batch sizes

### GPU not detected
```bash
# Check if NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi
```

### Memory limit exceeded
- Reduce memory limits in docker-compose files
- Monitor system with `htop` or `docker stats`
- Use smaller models or batch sizes

## Performance Tips

1. **GPU Training**: Use docker-compose.gpu.yml
2. **Large Datasets**: Use docker-compose.highmem.yml
3. **Limited RAM**: Reduce batch_size and num_workers
4. **Multiple GPUs**: Modify CUDA_VISIBLE_DEVICES environment variable
