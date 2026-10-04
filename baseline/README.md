# Baseline snapshot

`baseline_results.json` is the archived **pre-fix** experiment run (8 seed sets, dataset with duplicated
rows and popularity capped at 68, Greedy ILD inflated by arbitrary tie-breaking). `run_pipeline.py` prints
the new results next to it so you can see what changed.

Known defects of this run: duplicate tracks, seed leakage, genre mis-mapping, popularity ceiling at 68,
DPP returning duplicates. **Do not cite it as a result** - deltas against it measure the effect of the data and
algorithm fixes, not an algorithm improvement. It has no MMR entries and no `_std` fields.
