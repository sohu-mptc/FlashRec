"""Length-normalized beam score (matches SGLang `_calculate_beam_score`)."""


def beam_score(
    cum_logprob: float,
    seq_len: int,
    length_penalty: float = 1.0,
) -> float:
    if seq_len <= 0:
        return float(cum_logprob)
    return float(cum_logprob) / (float(seq_len) ** float(length_penalty))
