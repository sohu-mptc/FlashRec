# API

The server is a single FastAPI app, started with `--serve`:

```bash
flashrec --serve --model-path /path/to/model --port 8000 \
  --sid-vocab-file data/catalogs/sid2pid_beamrec_l4.json
```

`--model-path` plus `--sid-vocab-file` is the recommended serving configuration: SID layout is
inferred from the tokenizer. Leave the catalog unset only for an unconstrained
smoke test.

It binds `127.0.0.1:8000` by default and has **no authentication**. Bind a
public address only on a trusted network or behind an authenticating proxy.

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness probe |
| `GET` | `/v1/models` | Reports the loaded checkpoint |
| `POST` | `/v1/chat/completions` | Beam search (OpenAI-compatible) |
| `GET` `POST` | `/start_profile` | Begin a `torch.profiler` capture |
| `GET` `POST` | `/stop_profile` | End the capture, write the trace |

## `POST /v1/chat/completions`

Beam search over a chat prompt. Each beam is returned as one `choices[]` entry,
ranked best-first by cumulative log-probability.

### Request

| Field | Type | Default | Meaning |
| --- | --- | --- | --- |
| `messages` | array | required | Chat messages; `content` is a string or a list of `{"type":"text","text":...}` parts |
| `n` | int | `1` | Beam width **and** number of candidates returned |
| `max_tokens` | int | server default | Generated SID steps. `max_completion_tokens` is accepted as an alias |
| `temperature` | float | `0.0` | `0` = deterministic top-*k*; `> 0` = Gumbel top-*k* without replacement |
| `stream` | bool | `false` | `true` returns server-sent events; `false` returns one JSON body |
| `model` | string | null | Echoed back; the server serves one checkpoint |
| `chat_template_kwargs` | object | `{}` | Extra kwargs for the tokenizer's chat template |

`n` is beam width, so the slot budget bounds it: wide beam needs
`--cuda-graph-max-bs` / `--batch-slots` raised to match. See
[Configuration](configuration.md).

```bash
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"..."}],
       "n":50,"max_tokens":5,"temperature":0}'
```

### Response

Standard chat-completion shape, plus one non-standard key per choice:

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1767225600,
  "model": "/path/to/model",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "<sid_a><sid_b><sid_c>"},
      "finish_reason": "stop",
      "sglext": {"sequence_score": -1.8342}
    }
  ],
  "usage": {
    "prompt_tokens": 2481,
    "completion_tokens": 5,
    "total_tokens": 2486,
    "prompt_tokens_details": {"cached_tokens": 2432}
  }
}
```

- `sglext.sequence_score` is the beam's cumulative log-probability after the
  length penalty. Under `temperature > 0` it reports the noise-free score, while
  sampling itself uses the Gumbel-perturbed one.
- `finish_reason` is `"stop"` when the beam completed a full SID, `"length"` when
  it hit `max_tokens`.
- `prompt_tokens_details.cached_tokens` is the radix prefix-cache hit, which is
  how you confirm prefix reuse is working across requests.
- Choices can be **fewer than `n`** when the trie runs out of valid
  continuations; the response then carries every legal beam, and catalog size
  sets that ceiling.

An off-the-shelf OpenAI client works unchanged; read `sglext` directly from the
JSON when you want the scores.

### Streaming

With `"stream": true` the server emits compact SSE: one content chunk carrying
all beams, then a usage chunk, then `data: [DONE]`. Beams are emitted whole,
since per-beam token deltas inflated the payload roughly 3× in benchmarking.

## Profiling

`/start_profile` and `/stop_profile` mirror `sglang.bench_serving --profile`, so
the same tooling works against this server. The trace directory comes from
`FLASHREC_TORCH_PROFILER_DIR`, or `output_dir` in the request body.

```bash
curl -s http://127.0.0.1:8000/start_profile
# ... drive traffic ...
curl -s http://127.0.0.1:8000/stop_profile
```

Body fields: `output_dir`, `start_step`, `num_steps`, `activities`,
`profile_by_stage`, `with_stack`, `record_shapes`, `profile_prefix`. Details in
[Configuration](configuration.md).

## Python API

For offline use, drive the engine in-process:

```python
from flashrec import BeamRecConfig, BeamRecEngine

engine = BeamRecEngine(
    BeamRecConfig(
        model_path="/path/to/model",
        sid_vocab_file="data/catalogs/sid2pid_beamrec_l4.json",
    )
)
result = engine.generate(prompt="...", n=32, max_tokens=5)

for seq in result.sequences:          # ranked best-first
    print(seq.beam_score, seq.text)
```

`engine.generate_many([...])` batches a list of requests built with
`engine.make_request(...)`. Runnable versions of both:
[Examples](../examples/).
