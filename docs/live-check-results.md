# Live end-to-end check — results

Output of `scripts/live_check.py` run against a real provider endpoint, to prove
the full pipeline (ingest → classify → build schema → guided-JSON extract →
verify) works against a live API, not just the offline fake-client wiring.

> The private endpoint is intentionally redacted (kept in `.env`, gitignored).

## 2026-06-24 — OpenAI-compatible (local vLLM)

- **Provider:** `openai-compat`
- **Model:** `Qwen/Qwen3.6-35B-A3B-FP8` (served locally via vLLM)
- **Endpoint:** `http://<vllm-host>:8000/v1` (redacted)
- **Document:** `examples/sample_invoice.txt`

```
→ provider=openai-compat model=Qwen/Qwen3.6-35B-A3B-FP8
  doc_type   : invoice
  fields     :
    - invoice_number: INV-2025-00842
    - vendor_name: ACME SUPPLY CO.
    - invoice_date: 2025-11-03
    - total_due: 750.6
    - currency: USD
  coverage   : 0.80
  ungrounded : ['currency']
  result     : PASS ✅
    [x] doc_type present
    [x] coverage > 0
    [x] known sample values   (--strict)
```

**Notes**
- Classification, extraction, and the `--strict` known-value checks
  (`invoice_number`, `total_due`) all passed.
- Coverage is 0.80 because the grounding gate flagged `currency`: the model
  returned `USD` but its evidence quote didn't verbatim-verify against the
  source. That's the hallucination gate working as designed — surfacing an
  unverified value rather than silently trusting it.
- The model is a reasoning model (emits a "thinking" trace in free-form mode),
  but the extract path uses schema-constrained (guided JSON) decoding, so output
  is forced straight into the schema.

### Reproduce

```bash
OPENAI_BASE_URL=http://<vllm-host>:8000/v1 \
  uv run python scripts/live_check.py \
  --provider openai-compat --model "Qwen/Qwen3.6-35B-A3B-FP8" --strict
```
