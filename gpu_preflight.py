"""GPU resource benchmark on artificial tensors, not cancer-model evaluation."""
from pathlib import Path
import sys
import json
import time
import subprocess

PROJECT = Path('/kaggle/working/mammo_project')
sys.path.insert(0, str(PROJECT))
subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', 'pydicom>=3,<4'], check=True)
import torch
from mammo_reliability.audit import environment, require, write_json
from mammo_reliability.model import MammographyModel, seed_everything

require(torch.cuda.is_available(), 'CUDA is not available in this notebook session.')
seed_everything(17)
device = torch.device('cuda:0')
model = MammographyModel(pretrained=True).to(device).train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda')
benchmarks = []
for n_images in [2, 4, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    images = torch.randn(n_images, 3, 1024, 768, device=device)
    slots = torch.arange(n_images, device=device) % 2
    elapsed = []
    try:
        for step in range(7):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits, views = model(images, slots, 1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
            require(torch.isfinite(loss).item(), 'Nonfinite benchmark loss.')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            if step >= 2:
                elapsed.append(time.perf_counter() - started)
        benchmarks.append({'images_per_breast': n_images, 'status': 'completed',
                           'mean_step_seconds': sum(elapsed) / len(elapsed),
                           'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024**3})
    except torch.cuda.OutOfMemoryError:
        benchmarks.append({'images_per_breast': n_images, 'status': 'out_of_memory'})
        optimizer.zero_grad(set_to_none=True)
    del images, slots

result = {'purpose': 'artificial_tensor_GPU_resource_benchmark_only', 'environment': environment(),
          'gpu_count': torch.cuda.device_count(), 'gpu_name': torch.cuda.get_device_name(0),
          'gpu_total_gib': torch.cuda.get_device_properties(0).total_memory / 1024**3,
          'resolution': [1024, 768], 'benchmarks': benchmarks,
          'timing_limit': 'Excludes image loading, preprocessing, validation, and checkpoint writes.'}
write_json('/kaggle/working/mammo_private/technical_preflight/gpu_preflight.json', result)
print('GPU_PREFLIGHT_JSON_BEGIN\n' + json.dumps(result, indent=2) + '\nGPU_PREFLIGHT_JSON_END', flush=True)
