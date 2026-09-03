# Security Policy

Report vulnerabilities **privately**. Do not open a public GitHub issue.

安全问题请走私下披露，不要开公开 issue。

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | yes |

## Reporting a vulnerability

Do **not** open a public GitHub issue.

1. Prefer GitHub **Privately report a vulnerability** on this repository
   ([open an advisory](https://github.com/sohu-mptc/FlashRec/security/advisories/new)).
2. If that form is unavailable, contact the maintainers through a confidential channel.

Include the affected version or git commit, a minimal reproduction, and the
impact. We aim to acknowledge reports within 24 hours and to ship a patch
release for confirmed issues as soon as a fix is ready.

## Known operational caveats

- The HTTP server has **no built-in authentication**. The default bind
  address is `127.0.0.1`. Only bind a public address on a trusted network
  or behind an authenticating proxy.
- Do not pass secrets on the command line or in `--system-prompt`. Prefer
  files with restricted permissions if a prompt must include internal text.
- Model checkpoints, SID catalogs, and tokenizer files are supplied by the
  operator; treat them as trusted input.
