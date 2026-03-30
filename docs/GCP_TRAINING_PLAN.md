# GCP Training Plan for LeanFormer

Complete guide to set up a Google Cloud Platform environment from scratch and train
the LeanFormer reasoning core on a spot L4 GPU instance.

**Estimated cost:** ~$10
**Estimated training time:** ~34 hours
**Instance:** g2-standard-4 (1x L4 24GB, 4 vCPUs, 16GB RAM) spot VM

---

## Phase 1: Create a GCP Account

1. Go to https://cloud.google.com and click "Get started for free"
2. Sign in with a Google account
3. Enter billing information (credit card required)
4. GCP gives **$300 in free credits for 90 days** — this training will cost ~$10,
   well within the free tier

### Create a project

1. Go to https://console.cloud.google.com
2. Click the project dropdown (top bar) → "New Project"
3. Name it `leanformer-training`
4. Note the Project ID (you'll need it)

### Enable billing

1. Go to Billing → Link the project to your billing account
2. The $300 free credit is automatically applied

### Enable the Compute Engine API

1. Go to https://console.cloud.google.com/apis/library/compute.googleapis.com
2. Click "Enable"
3. Wait ~1-2 minutes for provisioning

---

## Phase 2: Install the gcloud CLI (Local Machine)

### Windows (your current setup)

```powershell
# Option A: Download the installer
# https://cloud.google.com/sdk/docs/install#windows
# Download and run GoogleCloudSDKInstaller.exe

# Option B: Via winget
winget install Google.CloudSDK
```

After installation, open a **new terminal** and run:

```bash
# Authenticate
gcloud auth login

# Set your project
gcloud config set project leanformer-training

# Set default region
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a
```

---

## Phase 3: Request GPU Quota

New GCP accounts start with **zero GPU quota**. You must request it.

1. Go to https://console.cloud.google.com/iam-admin/quotas
2. Filter by:
   - Service: "Compute Engine API"
   - Search for: "NVIDIA L4"
   - Region: "us-central1"
3. Find **"NVIDIA L4 GPUs"** — check the box
4. Click "Edit Quotas" at the top
5. Request a limit of **1**
6. In the justification, write: "ML model training — single L4 GPU for a ~66M parameter transformer, estimated 34 hours"
7. Submit

**Also request spot GPU quota:**
- Search for "Preemptible NVIDIA L4 GPUs" or "Spot NVIDIA L4 GPUs"
- Request limit of **1** with the same justification

**Turnaround:** Usually approved within minutes for small requests (1 GPU).
Occasionally takes up to 24-48 hours. If denied, try us-east1 or us-west1 instead.

**Fallback:** If L4 quota is delayed, request a T4 quota instead
(search "NVIDIA T4 GPUs"). T4 is more widely available and still
3x cheaper than local. Training will take ~49 hours instead of ~34.

---

## Phase 4: Create a Cloud Storage Bucket for Data

The training data is ~3GB. Upload it to GCS so the VM can pull it quickly.

```bash
# Create a bucket (name must be globally unique)
gcloud storage buckets create gs://leanformer-training-data \
  --location=us-central1

# Upload the tokenized training data
gcloud storage cp -r data/reasoning-core-tokenized/ \
  gs://leanformer-training-data/reasoning-core-tokenized/

# Upload the config
gcloud storage cp configs/reasoning_core.yaml \
  gs://leanformer-training-data/reasoning_core.yaml
```

---

## Phase 5: Create the GPU VM

```bash
gcloud compute instances create leanformer-trainer \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True" \
  --scopes=storage-ro \
  --maintenance-policy=TERMINATE
```

**What this does:**
- `g2-standard-4`: 4 vCPUs, 16GB RAM, 1x L4 24GB GPU
- `--provisioning-model=SPOT`: Spot pricing (~$0.28/hr instead of $0.71/hr)
- `--instance-termination-action=STOP`: If preempted, stops instead of deletes (preserves disk)
- `--image-family=pytorch-latest-gpu`: Pre-installed PyTorch, CUDA, cuDNN
- `--scopes=storage-ro`: Allows reading from GCS bucket
- `--boot-disk-size=100GB`: Room for code, data, and checkpoints

---

## Phase 6: Connect and Set Up the Environment

### SSH into the instance

```bash
gcloud compute ssh leanformer-trainer --zone=us-central1-a
```

### On the VM: Clone the repo and install

```bash
# Clone from GitHub
git clone https://github.com/brian-c-moore/LeanFormer.git
cd LeanFormer

# Install the package
pip install -e ".[dev]"

# Verify GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expected: NVIDIA L4
```

### Pull the training data from GCS

```bash
mkdir -p data/reasoning-core-tokenized
gcloud storage cp -r gs://leanformer-training-data/reasoning-core-tokenized/ \
  data/reasoning-core-tokenized/
```

### Verify everything

```bash
# Quick test — should see model build and a few training steps
python -c "
import torch, yaml
from leanformer.model.config import LeanFormerConfig
from leanformer.model.leanformer import LeanFormer

with open('configs/reasoning_core.yaml') as f:
    cfg = yaml.safe_load(f)
model = LeanFormer(LeanFormerConfig(**cfg['model'])).cuda()
x = torch.randint(0, 50257, (4, 512)).cuda()
out = model(x, labels=x, training=True)
print(f'Loss: {out[\"loss\"].item():.4f}')
print(f'VRAM: {torch.cuda.max_memory_allocated()/1e9:.1f} GB')
print('GPU setup verified.')
"
```

---

## Phase 7: Optimize Config for L4 (24GB VRAM)

The L4 has 24GB VRAM vs the RTX 3060's 12GB. We can increase the batch size
for better GPU utilization.

```bash
# Create an L4-optimized config
cp configs/reasoning_core.yaml configs/reasoning_core_l4.yaml
```

Edit `configs/reasoning_core_l4.yaml` — change only the training section:

```yaml
training:
  output_dir: ./checkpoints/reasoning_core
  epochs: 3
  batch_size: 4              # Doubled from 2 (L4 has 24GB VRAM)
  gradient_accumulation: 16  # Halved from 32 (keeps effective batch = 64)
  lr: 3.0e-4
  warmup_steps: 2000
  weight_decay: 0.01
  max_grad_norm: 1.0
  fp16: true
  logging_steps: 100
  save_steps: 5000
  eval_steps: 1000
  run_name: leanformer-reasoning-core-l4
```

Then update the config path in the training script temporarily:

```bash
# Or just copy the L4 config over the default
cp configs/reasoning_core_l4.yaml configs/reasoning_core.yaml
```

---

## Phase 8: Start Training

Always run long training inside `tmux` so it survives SSH disconnects.

```bash
# Start tmux session
tmux new -s training

# Start training
python -m leanformer.scripts.train_reasoning 2>&1 | tee training.log

# Detach from tmux: press Ctrl+B, then D
# Reattach later: tmux attach -t training
```

### Monitor from your local machine

```bash
# SSH in and check the log
gcloud compute ssh leanformer-trainer --zone=us-central1-a \
  --command="tail -20 /home/$USER/LeanFormer/training.log"
```

---

## Phase 9: Handle Spot Preemption

Spot VMs can be terminated with 30 seconds notice. The training script saves
checkpoints every 5,000 steps (~50 minutes on L4), so worst case you lose
~50 minutes of work.

### If preempted (VM stops):

```bash
# Restart the stopped VM
gcloud compute instances start leanformer-trainer --zone=us-central1-a

# SSH back in
gcloud compute ssh leanformer-trainer --zone=us-central1-a

# Check last checkpoint
ls -lt checkpoints/reasoning_core/step-*/

# Resume training — the script starts from step 0, but the best checkpoint
# is saved. To resume, you would need to modify the training script to
# accept a --resume flag, or simply note the best checkpoint and
# re-run with fewer remaining epochs.
```

**Important:** The current training script does not support automatic resume.
If preempted, the simplest approach is:
1. Check the best checkpoint saved so far
2. Note the val_loss — if it's already good (< 4.2), you may not need to continue
3. If more training is needed, the script will retrain from scratch but converge
   faster if initialized from the checkpoint (requires a small code change)

---

## Phase 10: Retrieve Results

### Download the checkpoint to your local machine

```bash
# From your local machine
gcloud compute scp --recurse \
  leanformer-trainer:~/LeanFormer/checkpoints/reasoning_core \
  checkpoints/reasoning_core \
  --zone=us-central1-a
```

### Or upload to GCS first (faster, resumable)

```bash
# On the VM
gcloud storage cp -r checkpoints/reasoning_core/ \
  gs://leanformer-training-data/checkpoints/reasoning_core/

# On your local machine
gcloud storage cp -r \
  gs://leanformer-training-data/checkpoints/reasoning_core/ \
  checkpoints/reasoning_core/
```

---

## Phase 11: Clean Up (Stop Billing)

**Do this immediately after retrieving results.**

```bash
# Delete the VM
gcloud compute instances delete leanformer-trainer --zone=us-central1-a

# Delete the GCS bucket (optional — storage cost is ~$0.02/month for 3GB)
gcloud storage rm -r gs://leanformer-training-data/

# Verify nothing is running
gcloud compute instances list
```

If you don't delete the VM, a stopped spot instance still incurs disk charges
(~$0.10/day for 100GB). A running instance costs ~$6.72/day on spot.

---

## Cost Summary

| Item | Cost |
|------|------|
| Compute (g2-standard-4 spot, ~34 hr) | ~$9.50 |
| Boot disk (100GB, ~2 days) | ~$0.20 |
| GCS storage (3GB, ~2 days) | ~$0.01 |
| Network egress (download checkpoint) | ~$0.10 |
| **Total** | **~$10** |

Covered entirely by the $300 free credit for new GCP accounts.

---

## Quick Reference: Commands You'll Run

```bash
# === ONE-TIME SETUP (local) ===
gcloud auth login
gcloud config set project leanformer-training
gcloud config set compute/region us-central1
gcloud config set compute/zone us-central1-a

# === UPLOAD DATA (local) ===
gcloud storage buckets create gs://leanformer-training-data --location=us-central1
gcloud storage cp -r data/reasoning-core-tokenized/ gs://leanformer-training-data/

# === CREATE VM (local) ===
gcloud compute instances create leanformer-trainer \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --metadata="install-nvidia-driver=True" \
  --scopes=storage-ro \
  --maintenance-policy=TERMINATE

# === SET UP VM (on VM) ===
gcloud compute ssh leanformer-trainer --zone=us-central1-a
git clone https://github.com/brian-c-moore/LeanFormer.git && cd LeanFormer
pip install -e ".[dev]"
mkdir -p data/reasoning-core-tokenized
gcloud storage cp -r gs://leanformer-training-data/reasoning-core-tokenized/ data/reasoning-core-tokenized/

# === TRAIN (on VM) ===
tmux new -s training
python -m leanformer.scripts.train_reasoning 2>&1 | tee training.log
# Ctrl+B, D to detach

# === MONITOR (local) ===
gcloud compute ssh leanformer-trainer --zone=us-central1-a --command="tail -20 ~/LeanFormer/training.log"

# === RETRIEVE (local) ===
gcloud compute scp --recurse leanformer-trainer:~/LeanFormer/checkpoints/reasoning_core checkpoints/reasoning_core --zone=us-central1-a

# === CLEAN UP (local) ===
gcloud compute instances delete leanformer-trainer --zone=us-central1-a
gcloud storage rm -r gs://leanformer-training-data/
```

---

## Troubleshooting

**"Quota exceeded" when creating VM:**
GPU quota hasn't been approved yet. Check status at
https://console.cloud.google.com/iam-admin/quotas. Try a different zone
(us-central1-b, us-central1-c) or region (us-east1-b).

**"ZONE_RESOURCE_POOL_EXHAUSTED" (spot not available):**
L4 spot capacity is temporarily exhausted in that zone. Try:
```bash
# Try other zones
gcloud compute instances create leanformer-trainer --zone=us-central1-b ...
gcloud compute instances create leanformer-trainer --zone=us-east1-b ...
```

**VM preempted during training:**
This is expected with spot VMs. Restart and check checkpoints (see Phase 9).

**CUDA out of memory with batch_size=4:**
Fall back to batch_size=2, gradient_accumulation=32 (same as local config).
The L4 should handle batch_size=4, but if the Deep Learning VM has different
PyTorch memory behavior, this is the safe fallback.

**Training loss is NaN:**
The script detects this and saves the log before exiting. Check
`checkpoints/reasoning_core/training_log.json` for the loss trajectory.
Usually caused by learning rate too high — try lr=1e-4.

**SSH connection drops:**
Training continues in tmux. Just reconnect:
```bash
gcloud compute ssh leanformer-trainer --zone=us-central1-a
tmux attach -t training
```
