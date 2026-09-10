# Online serving throughput

Wide-beam concurrent load against a running FlashRec server
(`POST /v1/chat/completions` with `n` = beam width).

Implementation: `python -m flashrec.benchmark.serving`
(requires `aiohttp`; install via `pip install 'flashrec[eval]'`).

```bash
# server
bash scripts/serve.sh   # or flashrec --serve ...

# client
python -m flashrec.benchmark.serving \
  --base-url http://127.0.0.1:8000 \
  --n 50 \
  --max-concurrency 32 \
  --num-prompts 200 \
  --warmup 8

# prompts from JSONL ({"prompt":"..."} or {"messages":[...]} per line)
python -m flashrec.benchmark.serving \
  --base-url http://127.0.0.1:8000 \
  --dataset-name file --dataset-path prompts.jsonl \
  --n 128 --max-concurrency 16

# torch.profiler via /start_profile /stop_profile
python -m flashrec.benchmark.serving \
  --base-url http://127.0.0.1:8000 \
  --n 50 --num-prompts 64 --max-concurrency 8 --profile
```

For RecIF recall / invalid_rate, use [`../recif/`](../recif/) instead.
