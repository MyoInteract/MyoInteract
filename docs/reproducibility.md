# Reproducible runs and regression benchmarks

Use Python 3.11 or 3.12 and `uv sync --frozen --extra test` for the locked CPU
environment. Add `--extra cuda` on supported Linux CUDA systems to install the
matching CUDA libraries through pip, or use `--extra cuda-local` with an existing
compatible CUDA/cuDNN toolkit (as in `env.yml`). Before first
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
Policy checkpoints under `checkpoints/` restore policy/value/normalizer
parameters. Loading them with `rl.load_checkpoint_path` is a
`parameter_warm_start`: it starts a new optimizer and training schedule.
`play_only` and environment evaluation mode are labeled `evaluation`.
Evaluation never writes to the source checkpoint directory or changes the
caller's configuration.

## Exact training continuation

New nonvision training runs also save full training-state snapshots under
`training_state/` when `rl.num_checkpoints` is positive. These snapshots retain
optimizer state, the global training step, RNG streams, and the running
environment state. Their integrity and complete configuration are checked
before restoration. Resume with the separate Hydra option
`rl.resume_training_path`; a run directory, its `training_state` directory,
or an individual completed step can be supplied. Root directories select the
latest completed snapshot and ignore unfinished writes.

For example, start a run with a total 15-million-step budget:

```bash
python -m myosuite.envs.myo.myouser.train_jax_ppo wandb=disabled run.suffix=initial rl.num_timesteps=15000000 rl.num_checkpoints=5
```

Continue from its most recent full snapshot into a separate output run:

```bash
python -m myosuite.envs.myo.myouser.train_jax_ppo wandb=disabled run.suffix=continued rl.num_timesteps=15000000 rl.num_checkpoints=5 rl.resume_training_path=/path/to/initial-run/training_state
```

Keep `rl.num_timesteps` at the **original total horizon**, not the number of
remaining steps or an increased budget. Keep the original environment, model
assets, network configuration, training hyperparameters, seed, dependency
versions, backend, and device topology. Mismatches fail explicitly. Logging
names and the destination run can change; source snapshots remain untouched.
The CLI examples above use the same default task configuration. For a custom
task, repeat all original Hydra overrides: selecting a snapshot does not
silently replace the requested configuration. Alternatively, load the
resolved configuration saved with that run in Python:

```python
from omegaconf import OmegaConf
from myosuite.envs.myo.myouser.train_jax_ppo import train

cfg = OmegaConf.load("/path/to/initial-run/checkpoints/config.json")
cfg.rl.load_checkpoint_path = None
cfg.rl.resume_training_path = "/path/to/initial-run/training_state"
cfg.run.suffix = "continued"
if cfg.wandb.enabled:
    cfg.wandb.name = "continued-run"
train(cfg)
```

The task fingerprint includes the effective environment configuration and a
hash of the compiled MuJoCo model, so changing included assets at the same
model path is detected.

Exact continuation cannot be combined with `rl.load_checkpoint_path`,
`run.play_only`, environment evaluation mode, or a zero training horizon.
Historical policy-only checkpoints cannot be used as full-state snapshots;
use `rl.load_checkpoint_path` for their parameter warm-start behavior. The
legacy observation migration applies only to policy loading and is never
used to bypass full-state compatibility checks. No new UI preset fields are
required: the existing UI checkpoint selector retains its warm-start meaning.

Full-state continuation currently supports nonvision training on a single
process with the same backend and topology. Madrona's external renderer state
has no verified snapshot contract, so vision runs retain policy checkpoints
and emit a warning instead of writing full-state snapshots. Explicit vision
resume is rejected. Fresh multi-process runs likewise retain policy-only
checkpoints and warn that exact full-state snapshots are unavailable.

CPU interruption tests compare the resumed and uninterrupted learner, optimizer,
normalizer, rollout state, RNG and logging counters bit for bit. Exact numerical
continuation requires deterministic kernels in the matching runtime. Elapsed
wall-clock measurements and external tracking/video side effects are outside
that comparison. Snapshots commit learner progress before evaluation/export
callbacks; resume does not replay those callbacks for the saved boundary.

The manifest labels this mode `exact_training_resume` and records the resolved
source path, source step, and content digest. Restoration flags remain false
while the run is only requested (`status: started`); successful restoration
and training mark optimizer, step, RNG, and environment restoration as verified.

The shipped mobile/public policies contain 54 input features although their
historical JSON declares 49. Their saved means, standard deviations and variance
accumulators prove that the final five target features were duplicated during
training. The loader recognizes only this exact legacy feature layout and
duplicate-statistics signature, reconstructs those target features on a private
configuration, and records the migration in the effective run configuration.
It preserves the policy weights and source checkpoint files. Other observation
shape mismatches fail explicitly rather than padding or truncating features.

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
