# Project notes for Claude Code

Mech-interp experiment template. See README.md for layout and extension points.

- Env: `uv` only (`uv run ...`, `uv add ...`). Never pip-install into the project env.
- vLLM lives in `envs/vllm` (separate env); don't add it to the main pyproject.
- Before finishing a change: `make check`; if models/ or sanity.py changed, also `make test-model`.
- New experiments go in `experiments/<name>/` (config.yaml + experiment.py subclassing `Experiment`);
  keep `src/interptemp` generic — no project-specific code there.
- nnsight rules: access modules in forward order inside a trace; create containers outside
  the `with` block; values escape only via `.save()`.
- Every intervention result needs a control; every LLM judge needs validation vs hand labels.

## Long-running commands (experiment runs, model loads, anything likely > ~30s)

The user must be able to watch progress and know how long to wait. So:

- Before launching: state a rough ETA and what it's based on (n items × est. time/item, device).
- Run in background (`run_in_background`), never as a blocking foreground call.
- Log the full raw output to a file; filter only when reading it back:
  `mkdir -p logs && PYTHONUNBUFFERED=1 uv run ... 2>&1 | tee logs/<name>-$(date +%Y%m%d-%H%M%S).log`
  No `grep`/`tail` in the pipe — they hide output until the process exits.
- Immediately give the user the log path and `tail -f logs/<file>.log` to follow it.
- Check the log periodically; report progress (done/total, elapsed, revised ETA) and stop early on errors.
- New experiment loops must report progress (`tqdm` over the main loop, or a log line per item with i/N).
