#!/usr/bin/env python3
"""Query a running FlashRec server over the OpenAI-compatible endpoint.

Start a server first (see examples/README.md), then:

    python examples/http_client.py --n 32
    python examples/http_client.py --url http://127.0.0.1:8000 --n 512 --top 10

Beams come back as `choices[]`, ranked best-first. Each choice carries its
cumulative log-prob under the non-standard `sglext.sequence_score` key; the rest
of the payload is plain OpenAI chat-completion shape, so an existing OpenAI
client works unchanged if you do not need the scores.

Uses only the standard library, so it runs without installing the package.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_PROMPT = (
    "The user has watched the following videos: "
    "<sid_0><sid_1><sid_2>. Recommend the next video."
)


def post_json(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--n", type=int, default=32, help="beam width / candidates")
    p.add_argument("--max-tokens", type=int, default=5, help="SID depth")
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="0 = deterministic top-k; >0 = Gumbel top-k without replacement",
    )
    p.add_argument("--top", type=int, default=5, help="candidates to print")
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    base = args.url.rstrip("/")
    payload = {
        "messages": [{"role": "user", "content": args.prompt}],
        "n": args.n,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    try:
        data = post_json(f"{base}/v1/chat/completions", payload, args.timeout)
    except urllib.error.URLError as exc:
        print(f"cannot reach {base}: {exc}", file=sys.stderr)
        print("is the server up? curl -s {}/health".format(base), file=sys.stderr)
        return 1

    choices = data.get("choices", [])
    usage = data.get("usage", {})
    print(f"model: {data.get('model')}")
    print(f"usage: {usage}")
    print(f"candidates: {len(choices)}\n")

    for choice in choices[: args.top]:
        score = (choice.get("sglext") or {}).get("sequence_score")
        text = choice.get("message", {}).get("content", "")
        print(
            f"[{choice.get('index')}] score={score} "
            f"finish={choice.get('finish_reason')} {text!r}"
        )

    if len(choices) < args.n:
        # Fewer beams than requested means the trie ran out of valid
        # continuations, not that the request failed.
        print(
            f"\nnote: asked for {args.n} candidates, got {len(choices)} "
            "(catalog exhausted the valid continuations)",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
