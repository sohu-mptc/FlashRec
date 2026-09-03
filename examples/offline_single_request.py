#!/usr/bin/env python3
"""Offline single-request beam search against a local checkpoint.

Runs the engine in-process: no server, no HTTP. Use this to sanity-check a
checkpoint and its SID catalog before putting a server in front of it.

    # Unconstrained (decodes over the full vocabulary):
    python examples/offline_single_request.py --model-path /path/to/OneRec-1.7B

    # Trie-constrained to a catalog (layout inferred from the tokenizer):
    python examples/offline_single_request.py \
      --model-path /path/to/OneRec-1.7B \
      --sid-vocab-file data/catalogs/sid2pid_beamrec_l4.json

See examples/README.md for where to download the checkpoint and catalog.
"""

from __future__ import annotations

import argparse
import logging

from flashrec import BeamRecConfig, BeamRecEngine

DEFAULT_PROMPT = (
    "The user has watched the following videos: "
    "<sid_0><sid_1><sid_2>. Recommend the next video."
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--beam-width", "--n", dest="beam_width", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=5)
    p.add_argument("--top", type=int, default=5, help="beams to print")
    # Catalog triggers tokenizer layout inference. Unset = full vocabulary.
    p.add_argument(
        "--sid-vocab-file",
        default=None,
        help="Valid-SID catalog (JSON). Layout is inferred from the tokenizer.",
    )
    p.add_argument(
        "--sid",
        default=None,
        help="Optional layout override START:END/SIZE,...",
    )
    p.add_argument("--sid-token-range", default=None)
    p.add_argument("--sid-codebook-sizes", default=None)
    p.add_argument("--sid-boundary-tokens", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    config = BeamRecConfig(
        model_path=args.model_path,
        beam_width=args.beam_width,
        max_tokens=args.max_tokens,
        sid=args.sid,
        sid_token_range=args.sid_token_range,
        sid_vocab_file=args.sid_vocab_file,
        sid_codebook_sizes=args.sid_codebook_sizes,
        sid_boundary_tokens=args.sid_boundary_tokens,
    )

    # Loads weights, allocates the KV pool, and captures CUDA graphs; the first
    # call is slow, steady-state cost is one graph replay per decode step.
    engine = BeamRecEngine(config)

    result = engine.generate(
        prompt=args.prompt,
        n=args.beam_width,
        max_tokens=args.max_tokens,
    )

    print(f"prompt tokens: {result.prompt_tokens}")
    print(f"completion tokens: {result.completion_tokens}")
    print(f"cached tokens: {result.cached_tokens}")
    print(f"beams returned: {len(result.sequences)}\n")

    # sequences are already ranked best-first by cumulative log-prob.
    for rank, seq in enumerate(result.sequences[: args.top]):
        print(f"[{rank}] score={seq.beam_score:.4f} {seq.text!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
