# Security Policy

## Supported versions

This is pre-alpha software. Only the latest `main` is supported; there are no
backported fixes.

## Reporting a vulnerability

Please report security issues privately via
[GitHub Security Advisories](https://github.com/Lawson-Darrow/rag-conflict-eval/security/advisories/new)
rather than a public issue. We'll acknowledge and respond as soon as we can.

## Scope notes

This tool sends your prompts, retrieved contexts, and candidate answers to whatever
LLM provider you configure via `litellm`. Do not run it on sensitive data with a
provider you don't trust. Untrusted source/answer text is treated as data (delimiter
tokens are neutralized in judge prompts), but prompt-injection defense in
LLM-as-judge setups is best-effort, not a guarantee.
