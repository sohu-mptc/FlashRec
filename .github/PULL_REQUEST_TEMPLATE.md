## Summary

<!-- 1-3 bullets: why this change exists, not a file list. -->

## Test plan

- [ ] `pre-commit run --all-files`
- [ ] `PYTHONPATH=python python -m unittest discover -s tests -v`
- [ ] New/changed CLI flags documented in `docs/configuration.md` and `docs/configuration.zh-CN.md`; HTTP fields in `docs/api.md`
- [ ] User-visible behaviour recorded under `Unreleased` in `CHANGELOG.md`

## Notes

<!-- CUDA-graph capture safety, SID layout (catalog + tokenizer inference), FP8 path, or anything a reviewer should try on GPU. -->
