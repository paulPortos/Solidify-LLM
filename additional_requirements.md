# Additional Requirements

## GPU Support (CUDA 12.8)

**PyTorch with CUDA support cannot be installed via requirements.txt**

Run this command separately after creating your virtual environment:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## Installation Order

1. Create and activate virtual environment
2. Install PyTorch with CUDA (command above)
3. Install other dependencies: `pip install -r requirements.txt`

## Note
- This ensures GPU acceleration for model training
- CPU-only version will work but will be significantly slower
- Verify CUDA installation with: `python -c "import torch;