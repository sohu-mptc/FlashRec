from eval_beam_matrix import parse_sglang_generate, uses_sglang_generate


class TestSglangProtocol:
    def test_auto_detects_sglang(self):
        assert uses_sglang_generate("sglang", "auto") is True
        assert uses_sglang_generate("flashrec", "auto") is False
        assert uses_sglang_generate("flashrec", "generate") is True
        assert uses_sglang_generate("sglang", "chat") is False

    def test_parse_beam_results(self):
        payload = {
            "text": "ignored",
            "meta_info": {
                "beam_results": [
                    {
                        "text": "<s_a_1><s_b_2><s_c_3>",
                        "meta_info": {"sequence_score": -1.5},
                    },
                    {"text": "not a sid", "meta_info": {}},
                ]
            },
        }
        out = parse_sglang_generate(payload, n=50)
        assert len(out) == 2
        assert out[0] == ("<s_a_1><s_b_2><s_c_3>", -1.5)
        assert out[1] == ("", None)

    def test_parse_plain_text_fallback(self):
        out = parse_sglang_generate({"text": "<s_a_0><s_b_0><s_c_1>"}, n=4)
        assert out == [("<s_a_0><s_b_0><s_c_1>", None)]
