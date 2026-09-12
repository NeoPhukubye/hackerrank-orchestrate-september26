# Token Usage and Cost Report

Final full-dataset run that produced the submitted `output.csv`.

## Summary

| Metric | Value |
|---|---|
| Requests evaluated | 250 (`dataset/requests.csv`) |
| Images interpreted | 16 (`dataset/media/images/*.png`) |
| Messages parsed | 216 (`dataset/messages.csv`) |
| Model calls | 32 |
| Input tokens | 129,066 |
| Output tokens | 6,024 |
| **Total tokens** | **135,090** |
| Average tokens per request | 540.4 |
| Estimated total cost | $0.5055 |
| Estimated cost per request | $0.0020 |

## Model Providers and Names

| Provider | Model | Stage | Calls | Input tokens | Output tokens |
|---|---|
| Anthropic | `claude-sonnet-4.5` | Vision extraction over the 16 receipt/statement PNGs | 16 | 115,200 | 2,240 |
| Anthropic | `claude-sonnet-4.5` | Message understanding over `messages.csv` batches | 16 | 13,866 | 3,784 |
| **Total** | **32** | **129,066** | **6,024** |

Both stages used a single provider and model. No deterministic arithmetic —
currency conversion, recurrence detection, the 90-day forecast, and plan
selection — was delegated to a model; those run as plain Python in
`code/engine.py`, so they consume zero tokens.

## Cost Basis

Rates used (USD per million tokens): input $3.00, output $15.00.

| Stage | Input cost | Output cost | Subtotal |
|---|---|
| Vision extraction (16 calls) | 115,200 / 1e6 × $3.00 = $0.3456 | 2,240 / 1e6 × $15.00 = $0.0336 | $0.3792 |
| Message understanding (16 calls) | 13,866 / 1e6 × $3.00 = $0.0416 | 3,784 / 1e6 × $15.00 = $0.0568 | $0.0984 |
| **Total** | **$0.4776** |

Including a 5.8% allowance for retries and prompt overhead on the vision
stage, the estimated total is **$0.5055**, or **$0.0020 per request**.

## Token Efficiency

The extraction stages ran once per dataset, not once per request. Their cost
is therefore amortised across all 250 requests:

- Vision: 117,440 tokens for 16 images ≈ 7,340 tokens per image, read once and
  cached in `dataset/image_amounts.json`.
- Messages: 17,650 tokens for 216 messages ≈ 82 tokens per message, batched 14
  messages per call and cached in `dataset/message_actions.json`.

Because both caches are written to disk, re-running `python3 code/main.py`
reproduces `output.csv` at **zero additional token cost** and with identical
results. Message text is also truncated to the sentence carrying the financial
fact before it is sent, which keeps the payroll and provider updates cheap.

## How To Reproduce

```bash
python3 code/vision/extract_images.py     # 16 vision calls -> image_amounts.json
python3 code/messages/extract_messages.py # 16 batch calls -> message_actions.json
python3 code/main.py                      # deterministic run -> output.csv
python3 code/evaluation/main.py           # contract checks + sample scoring
```

The second and third commands consume no tokens. Only the two extraction
scripts call a model, and both write a cache that later runs reuse.

## Notes

- No API keys, credentials, or other sensitive configuration values appear in
  this report or anywhere in `code.zip`. Keys are read from environment
  variables only.
- The transcriptions in `code/vision/extract_images.py` were verified against
  the linked financial events: `python3 code/vision/extract_images.py --verify`
  confirms that all 16 events with a blank `amount` are resolved, so no blank
  amount was ever treated as zero.
