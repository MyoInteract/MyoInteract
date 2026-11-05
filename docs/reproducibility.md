# Reproducible runs and regression benchmarks

Use Python 3.11 or 3.12 and `uv sync --frozen --extra test` for the locked CPU
environment. Add `--extra cuda` on supported Linux CUDA systems. Before first
import in a fresh test environment, run `uv run python scripts/prepare_test_assets.py`.
This prepares the exact Menagerie revision requested by the pinned Playground
package, with only metadata checked out; the MyoInteract models ship here.

Training writes `run_manifest.json` beside its checkpoints. The manifest records
the effective configuration and its SHA-256 digest, Git revision and dirty state
when running from a source checkout, dependency versions, platform, backend,
devices, and training/evaluation seeds. Installed wheels report no Git revision
rather than borrowing the revision of an unrelated enclosing project.

When loading a checkpoint, the manifest identifies the resolved numbered step
and hashes its filenames and bytes, so policy identity survives relocation.
The current checkpoint format restores policy/value/normalizer parameters. It
does **not** restore optimizer state or the original training step count. A run
that trains after loading is labeled `parameter_warm_start`; `play_only` and
environment evaluation mode are labeled `evaluation`. Evaluation never writes
to the source checkpoint directory or changes the caller's configuration.

The benchmark covers `ar`, `mobile-typing-5c-words`, and
`public-display-4-buttons`. Their reference task configurations are bundled with
the package. With a source checkout's tracked policy files, run:

```bash
uv run myointeract-benchmark --seeds 0 1 2 --episodes 3 --output benchmark-results/reference.json
```

Installed wheels omit the large policy files. Supply
`--checkpoints-dir /path/to/tracked_checkpoints/universal`, or use `--policy zero`
for a control/environment regression. A bounded CPU smoke protocol is:

```bash
uv run myointeract-benchmark --policy zero --seeds 0 --episodes 1 --max-steps 2 --output benchmark-results/cpu-smoke.json
```

`--max-steps` truncates evaluation for smoke testing; these short results do not
measure task proficiency. Omit it for full episodes. The output contains protocol,
per-case configuration/provenance, episode success/time, returns, and reward
components. The benchmark fails on non-finite rewards or a mismatch between the
optimized reward and its reported components. It also detects the lifecycle,
acceleration-cost, and completion-metric defects through the accompanying
focused regression tests.

Compare like-for-like runs explicitly:

```bash
uv run myointeract-benchmark --policy zero --seeds 0 --episodes 1 --max-steps 2 --compare benchmark-results/cpu-smoke.json --output benchmark-results/comparison.json
```

The comparator requires matching tasks, configuration digests, policy checkpoint
steps and content hashes, seeds, episode counts, and step limits. Return/component tolerances default
to absolute and relative `1e-4`, with no success-rate change allowed. Override
`--reward-atol`, `--reward-rtol`, `--success-atol`, and `--duration-atol` for a
justified protocol. Mean duration defaults to a `1e-6` second tolerance.
The comparison records these tolerances with its results.

When the movement metrics from [PR #3](https://github.com/MyoInteract/MyoInteract/pull/3)
are installed, the benchmark calls their `rollout_movement_metrics` adapter and
includes its output. Otherwise it records that the optional metrics are
unavailable. It does not duplicate the scientific formulas. Reset samples are
included for movement trajectories and excluded from reward aggregation.

Reward and reset corrections change training semantics. Existing tracked
policies are historical policies; evaluating them on corrected code is not
retraining them. Archive full corrected evaluations and newly trained policies
under separate run manifests before using them as replacement research baselines.

Normal CI runs real CPU regression tests, including a custom PPO update and
checkpoint round trip. A manual workflow dispatch additionally evaluates the
three-task bounded smoke protocol and uploads its JSON. Real Madrona GPU
execution requires a dedicated supported Linux/CUDA runner and is not claimed
by renderer-contract tests that use a mock renderer.
