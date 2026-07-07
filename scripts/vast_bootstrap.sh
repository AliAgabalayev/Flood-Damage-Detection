#!/usr/bin/env bash
set -euo pipefail

echo ">> GPU check"
nvidia-smi

echo ">> Python environment"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ">> Confirming CUDA is visible to PyTorch"
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo ">> torch is CPU-only, reinstalling from the CUDA index (known requirements.txt issue)"
  pip install -q torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu121
fi
python -c "import torch; assert torch.cuda.is_available(), 'CUDA still not available to PyTorch after reinstall!'; print(torch.cuda.get_device_name(0))"

echo ">> Downloading Sen1Floods11 data (hand + weak)"
bash scripts/download_data.sh
bash scripts/download_weak_data.sh

echo ">> Validating config"
make config

echo ">> Environment ready. Launch whichever training job you want, e.g.:"
echo "     mkdir -p logs && nohup make pretrain-finetune > logs/pretrain_finetune.log 2>&1 &"
