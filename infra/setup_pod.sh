#!/usr/bin/env bash
# Bootstrap a fresh GPU box (Sesterce / Runpod / Vast). Idempotent; re-run after restarts.
#   git clone https://github.com/<user>/<repo>.git && cd <repo> && WITH_VLLM=1 bash infra/setup_pod.sh
# Env knobs: PERSIST (cache root; auto-detected), MODEL (weights to prefetch), WITH_VLLM=1.
set -euo pipefail

MIN_DRIVER_CUDA="12.6"   # torch is pinned to cu126 (main env) / cu129 (vLLM env), see pyproject

die() { echo "!! $*" >&2; exit 1; }

# Big disk for caches: a mounted volume if present, else the provider's scratch disk, else $HOME.
pick_persist() {
  for d in /workspace /ephemeral; do
    [ -d "$d" ] && [ -w "$d" ] && { echo "$d"; return; }
  done
  echo "$HOME"
}
PERSIST="${PERSIST:-$(pick_persist)}"
export HF_HOME="${HF_HOME:-$PERSIST/hf}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PERSIST/uv-cache}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"
echo ">> caches under $PERSIST ($(df -h "$PERSIST" | awk 'NR==2 {print $4}') free)"

# Fail early on an old driver: CUDA wheels newer than the driver import fine but die on first use.
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
driver_cuda=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+')
[ "$(printf '%s\n' "$MIN_DRIVER_CUDA" "$driver_cuda" | sort -V | head -1)" = "$MIN_DRIVER_CUDA" ] \
  || die "driver supports CUDA $driver_cuda < $MIN_DRIVER_CUDA: pick an image with a newer driver"

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# Persist env vars for new shells; the marked block is rewritten so PERSIST changes stick.
sed -i '/# >>> setup_pod >>>/,/# <<< setup_pod <<</d' ~/.bashrc 2>/dev/null || true
cat >> ~/.bashrc <<RC
# >>> setup_pod >>>
export HF_HOME=$HF_HOME
export UV_CACHE_DIR=$UV_CACHE_DIR
export PATH="\$HOME/.local/bin:\$PATH"
# <<< setup_pod <<<
RC

uv sync --locked
[ -f .env ] || { cp .env.example .env; echo ">> fill in .env (OPENROUTER_API_KEY, HF_TOKEN)"; }
set -a; source .env; set +a

# Run a real kernel: torch.cuda.is_available() can be True while kernels still fail.
check_cuda() {
  uv run "$@" python - <<'PY'
import torch
x = torch.ones(2, device="cuda")
assert (x * 2).sum().item() == 4.0
print(f"   torch {torch.__version__} (CUDA {torch.version.cuda}) on {torch.cuda.get_device_name(0)}: ok")
PY
}
echo ">> CUDA check (main env)"
check_cuda || die "CUDA kernels fail in the main env (driver vs torch CUDA mismatch?)"

uv run hf download "$MODEL" --exclude "*.pth" --exclude "original/*" >/dev/null
echo ">> $MODEL cached in $HF_HOME"

if [ "${WITH_VLLM:-0}" = "1" ]; then
  uv sync --project envs/vllm --locked
  echo ">> CUDA check (vLLM env)"
  check_cuda --project envs/vllm || die "CUDA kernels fail in the vLLM env"
fi

command -v tmux >/dev/null || echo ">> tmux missing: install it, long runs must survive SSH drops"
echo ">> next: make check && make sanity   (open a new shell or 'source ~/.bashrc' first)"
