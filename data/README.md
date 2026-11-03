# Data

`sample_20.jsonl` — the first 20 English (`language == "en"`) rows of the
`train` split, created with:

```bash
uv run python scripts/fetch_sample.py
```

One flat JSON object per line: `row_idx`, `subject`, `body`, `answer`, `type`,
`queue`, `priority`, `language`, `tag_1` … `tag_8`.

## Source and license

[Tobi-Bueck/customer-support-tickets](https://huggingface.co/datasets/Tobi-Bueck/customer-support-tickets)
on Hugging Face, licensed CC BY-NC 4.0. Used here for non-commercial learning only.
