MyoInteract project review and proposed pull requests

Reviewed on 23 October 2026, against local `master` commit `b67d94bb3676a272d080126a097218ecf39bf052`.

MyoInteract has a coherent research-prototype core: configurable pointing and button tasks, vectorized JAX training, three supplied presets, and tracked policy checkpoints. The main weakness is the correctness and verification of the task/training lifecycle. Several defects can change the reward optimized by PPO, carry task state between episodes, or misreport evaluation outcomes. These deserve priority over new features or performance tuning.

The default setup and automation also lag the implementation: the documented Colab workflow changes out of the checkout while models and presets use relative paths; CI targets different branches from the repository's default branch; and the copied custom PPO tests import upstream Brax rather than the modified implementation. These gaps make regressions difficult to detect.

The findings below establish problems in this checkout. They do not establish which historical experiments or published results used the affected paths. Changes to rewards or episode semantics should trigger fresh evaluation and, where applicable, retraining with the corrected implementation.

**Scope and validation**

- Reviewed the MyoInteract environment, target lifecycle, Gradio configuration workflow, notebook setup, custom training/evaluation/checkpoint wiring, package metadata, and CI. This was not an exhaustive audit of every inherited MyoSuite or vendored User-in-the-Box component.
- All 162 locally present Python source files under `myosuite` parsed successfully. Parsing is not a runtime test.
- Reproductions used unchanged methods/callbacks extracted through Python AST, small test environments, temporary directories, and direct MuJoCo model loading. The actual arm XML compiled successfully.
- Installed JAX is 0.9.2 and MuJoCo is 3.6.0; the project pins JAX below 0.6.0 and MuJoCo to 3.3.5. These isolated checks are evidence of source-level defects, not validation of the pinned stack.
- Brax, MuJoCo Playground, and Gradio are absent locally. Full PPO training, GPU execution, renderer integration, and a full UI session were not run. Integration acceptance checks are specified per proposal.
- At the time of the initial review, no implementation changes or remote PRs had been created. The subsequent implementation and verification are mapped in [IMPLEMENTATION.md](IMPLEMENTATION.md).

P1 means fix before relying on new training runs or the advertised workflow. P2 means a concrete correctness, reliability, or reproducibility fix. P3 means an improvement to plan after the urgent fixes.

**Proposed bug-fix PRs, in recommended order**

1. **P1: Preserve terminal rewards and fresh episode state during auto-reset.**

   Evidence: `myosuite/train/utils/wrapper.py:315-318` obtains a new reset state, then overwrites its entire `info` with terminal information. It does not restore the terminal reward, so the reset reward of zero reaches PPO. The environment reset at `myosuite/envs/myo/myouser/myouser_universal.py:556-594` explicitly resets phase, dwell progress, task flags, RNG, and targets, which this wrapper discards.

   Isolated execution of the current wrapper produced terminal reward `[0.]` instead of `[7.]`, reset data/observations with stale `phase=[2]` instead of zero, and a discarded reset RNG advance.

   Scope: preserve terminal reward/done/metrics while retaining fresh environment reset information. Carry only explicitly identified wrapper bookkeeping across the reset. Review episode metric/counter semantics at both completion and time limits.

   Acceptance: under JIT and vectorization, run multiple completed and truncated episodes; terminal rewards remain intact, each episode starts at phase zero with cleared dwell/task flags, RNG advances, targets and observations agree, and wrapper counters remain correct.

2. **P1: Fix acceleration effort reward indexing.**

   Evidence: `myosuite/envs/myo/myouser/myouser_universal.py:610` already filters accelerations to independent degrees of freedom. Line 674 indexes that reduced array using the original full-model indices again. The actual model has independent DOFs `[11,12,14,15,16]`, while the reduced array has length five. JAX clamps these reads to the final entry, ignoring the other four joints.

   Isolated execution with reduced acceleration `[100,0,0,0,0]` and zero controls returned effort cost `0.0`; the intended formula gives `0.0667`. The tracked AR configuration enables this effort term.

   Scope: compute the acceleration norm from the already-filtered observation, with one clearly defined indexing boundary.

   Acceptance: excite each independent joint separately; compare against full simulation acceleration filtered exactly once. Each joint contributes the expected squared acceleration. Re-evaluate affected reward comparisons after the fix.

3. **P1: Bound target placement sampling and reject infeasible geometry.**

   Evidence: `myosuite/envs/myo/myouser/myouser_universal.py:291,304-308` uses an unbounded rejection loop requiring the next position to be at least twice its maximum radius from the previous target. UI validation at `gradio_ui.py:472-483` permits two spheres fixed at the same point or confined to an impossibly small region.

   With both positions fixed at `[0.3,0,0]` and radius `0.1`, every sample is the same point and the loop condition remains true forever. The sampler and continuation predicate were checked individually; the infinite loop was not launched.

   Scope: validate configurations before launching training, bound rejection attempts, and define an explicit failure or fallback behavior that works under JIT. Do not silently alter the task's separation requirement.

   Acceptance: identical fixed targets and infeasible narrow ranges fail promptly with an actionable message. Valid ranges produce deterministic seeded placements and bounded reset execution under JIT/vectorization.

4. **P1: Resolve bundled resources and run directories independently of the working directory.**

   Evidence: the Colab setup cell in `myointeract_ui.ipynb` installs inside the clone and then restores the original directory. `UniversalEnvConfig.model_path` at `myouser_universal.py:193` is relative, and `base.py:109-110,137` passes it directly to MuJoCo. Loading from another directory reproduced a model-file opening error.

   Presets are also cwd-relative at `gradio_ui.py:799-828`. Meanwhile, the UI discovers repository-root logs at `gradio_ui.py:35-37`, but training writes cwd-relative logs at `train_jax_ppo.py:106`. The three advertised presets can disappear and newly trained checkpoints can be undiscoverable.

   Scope: resolve bundled model/preset resources from an explicit package/resource base and use one configurable writable output directory across UI, training, and checkpoint discovery. Preserve support for explicit user model paths.

   Acceptance: start from outside the checkout, load each shipped preset, render a task, complete a small training run, and find its checkpoint in the UI. Test both the notebook workflow and installed-package resource availability.

5. **P2: Restore CI coverage for this repository and test the custom implementation.**

   Evidence: `.github/workflows/python-app.yml:7,13` only triggers on `main` and `dev`, while GitHub reports the default branch as `master`. The workflow uses Python 3.9 at line 52, but pinned Brax 0.12.4 and Orbax Checkpoint 0.11.21 both require Python >=3.10. `pyproject.toml:7` still advertises >=3.9; `env.yml` uses 3.11.

   The copied tests at `myosuite/train/myouser/custom_ppo/train_test.py:26-28` and `checkpoint_test.py:23-25` import upstream Brax trainer/checkpoint modules. They do not protect the project's modified implementations. The release workflow additionally invokes nonexistent `setup.py` at `.github/workflows/pipy_release.yml:114`.

   Scope: trigger checks for the active branch, align the supported Python version with a tested environment, run project-specific CPU smoke/regression tests, and use a working build command such as `python -m build`. Keep any GPU checks in a separate job appropriate to available runners.

   Acceptance: a PR targeting `master` starts CI; a clean environment installs successfully; a short custom PPO/checkpoint round trip executes project modules; a distribution builds; regression tests from PRs 1-4 fail on the old implementation and pass after their fixes.

   Package metadata references: [Brax 0.12.4](https://pypi.org/pypi/brax/0.12.4/json) and [Orbax Checkpoint 0.11.21](https://pypi.org/pypi/orbax-checkpoint/0.11.21/json).

6. **P2: Repair single-episode evaluation and final-target success reporting.**

   Evidence: `myosuite/train/utils/wrapper.py:383-394` splits reset keys only when batch size exceeds one, then always applies `vmap`. `evaluate_policy` defaults to one episode. The current predefined reset produced a batch-size mismatch between `rng uint32[2]` and `eval_id int32[1]`; the random-reset fallback received an invalid scalar PRNG key.

   Separately, `myouser_universal.py:655-659` looks for `target_N_completed` for every target, while the final target emits `target_final_completed` at line 744. A successful two-target rollout reported `[True,False]`; successful single-target tasks also appear unsuccessful in this per-target evaluation output.

   Scope: always provide a batch of reset keys and unify or explicitly translate the completion metric contract. Coordinate with open PR #3, which changes evaluation and adds movement metrics.

   Acceptance: seeded evaluation works with one, two, and several episodes, including random fallback. Successful single- and multiple-target episodes report correct per-target success; incomplete episodes preserve partial completion.

7. **P2: Keep checkpoint loading and evaluation read-only.**

   Evidence: `myosuite/train/utils/train.py:352-353` mutates `config.rl.num_timesteps` to zero for evaluation. When no output directory is supplied, lines 364-392 retain the source checkpoint path and unconditionally write configuration there. `evaluate.py:101` invokes that path.

   Running the current setup statements against a temporary checkpoint changed its recorded training budget from `15000000` to `0`.

   Scope: separate input checkpoint paths from output paths and copy configuration before applying evaluation overrides. Use a single path resolver for checkpoints roots and individual step directories.

   Acceptance: evaluation leaves both source-file hashes and caller-supplied configuration unchanged; roots and individual-step paths load consistently; read-only checkpoints can be evaluated.

8. **P2: Confine preset saves to their directory and preserve configuration names.**

   Evidence: `myosuite/envs/myo/myouser/gradio_ui.py:809-820` interpolates unchecked textbox content into a filename opened with `"w"`. In a temporary directory, saving `../escaped` created JSON outside `gradio_configs`. This can overwrite writable JSON files if a caller supplies an existing path. Risk increases if the UI is shared; the standalone entry point enables sharing at line 1274.

   Discovery at line 803 also splits filenames at the first dot. Saving `foo.bar` creates `foo.bar.json`, but after restarting, the UI advertises `foo` and loading it raises `FileNotFoundError`.

   Scope: validate names, resolve destinations within the configured directory, reject traversal/separators, use exclusive creation where overwrites are disallowed, and use the complete filename stem for discovery.

   Acceptance: traversal and absolute-path inputs cause no outside writes; existing files remain intact; ordinary and dotted names round-trip after restart.

9. **P2: Reject invalid UI configurations before accepting Run.**

   Evidence: `gradio_ui.py:1079-1080` ignores the biomechanical validation result. A shipped preset with `ctrl_dt=None` produced a failed biomechanical check but an overall successful configuration, silently omitting the requested timestep override.

   RL fields allow zero at lines 698, 705, and 712; validation at 751-762 only checks whether `int(value)` succeeds. It accepts `num_envs=0`, fractional values such as `1.9`, and nondivisible dimensions such as batch_size=128, num_envs=1000, num_minibatches=8. The trainer at `custom_ppo/train.py:331` then divides by zero or raises an assertion.

   Scope: propagate every validation result and enforce finite values, actual integers, required positivity, valid timestep/dwell relationships, and training dimension constraints in a shared validator used by UI and CLI.

   Acceptance: invalid input produces specific messages and leaves RunState unchanged; all supplied presets pass; accepted dimensions reach a training update successfully.

10. **P2: Keep dwell progress local to each target phase.**

    Evidence: `myouser_universal.py:352` increments the previous phase's `steps_inside_target`, and `update_info` at 631-635 carries it across phase changes. Overlapping successive targets can therefore complete without their own dwell interval. Overlap is permitted by separation based only on the next target's radius at line 291.

    Isolated execution with five prior dwell steps and a next target requiring five steps completed that target on its first inside step, with counter six. Transition observations at lines 355-360 also retain the completed target's position/size while reporting the next phase.

    Scope: clear dwell state on phase transitions and define consistent next-phase target observations.

    Acceptance: overlapping sphere targets and button-to-sphere transitions each require the full configured dwell for the new target; phase and target observations agree at transition boundaries.

11. **P2: Make checkpoint logging honor disabled W&B.**

    Evidence: `train/utils/train.py:524-535` constructs the default evaluation checkpoint callback even with W&B disabled. It unconditionally calls plotting at line 246 and `wandb.log` at 274; the plot functions also call W&B. A disabled tracking run can fail at its first checkpoint or log to an unrelated existing global run.

    Scope: make the remote logging decision explicit throughout callbacks and separate local checkpoint persistence/evaluation from optional remote reporting.

    Acceptance: a short run with tracking disabled saves final checkpoints while W&B calls are forbidden by mocks. Enabled mode retains its metrics and artifact behavior.

12. **P2: Supply a valid evaluation environment for vision training callbacks.**

    Evidence: `train/utils/train.py:467-470` sets `eval_env=None` for vision. Both video callback setup at 511-519 and default checkpoint setup at 524-535 can instantiate `ProgressEvalLogger` with it. The initializer at line 196 immediately calls a helper that dereferences `prepare_eval_rollout`, so ordinary vision training fails before an update. Loading through `evaluate.py:101` also fails to forward the stored vision setting.

    Scope: use the existing renderer-backed environment for compatible callbacks, define its batch/reset contract, and forward vision configuration during loading. Avoid allocating a second renderer unnecessarily.

    Acceptance: on a supported GPU/renderer setup, complete a small vision update, save/reload a checkpoint, and evaluate it with correct fixed batch dimensions. Document and test this path separately from nonvision CPU tests.

**Improvement PRs to follow the correctness fixes**

13. **P3: Introduce versioned named-field presets and run manifests.**

    The UI presets are positional arrays tied to widget order, while checkpoints record configuration without a complete reproducibility manifest. Add a schema version and named fields, migrate all three supplied presets, and record commit, effective configuration, package versions, backend/device details, and seeds with each run. Explicitly distinguish a parameter-only warm start from full training continuation; the current inherited checkpoint behavior does not preserve optimizer state and training step count.

    Acceptance: old presets migrate without semantic changes; malformed/unsupported versions fail clearly; manifests make a checkpoint's task and runtime identifiable without relying on its directory name.

14. **P3: Add a small scientific regression benchmark using the three shipped tasks.**

    After fixing lifecycle/reward semantics, establish reference evaluations over fixed seeds, including success, task time, reward components, and movement metrics where appropriate. Build on open PR #3 rather than duplicating its metrics implementation. Compare reward components against explicit formulas and track acceptable variation across seeds. Archive the corrected baselines separately from historical checkpoints.

    Acceptance: the benchmark detects known reset/reward/evaluation regressions, produces machine-readable results, and reports enough context to compare runs. Use fast CPU checks for ordinary PRs and a bounded GPU benchmark where resources permit.

15. **P3: Make previews reproducible and distinguish rendering from resampling.**

    `gradio_ui.py:1135-1146` renders using seed N, then changes the hidden seed to N+1. A subsequent saved configuration therefore describes a different initial sphere layout. Keep rendering free of configuration changes or expose a separate Resample action which increments before rendering.

    Acceptance: saving immediately after a preview and reopening that configuration recreates the displayed initial geometry.

**Existing PRs to coordinate with**

- [#1, motor noise](https://github.com/MyoInteract/MyoInteract/pull/1)
- [#2, fatigue model](https://github.com/MyoInteract/MyoInteract/pull/2)
- [#3, movement quality metrics](https://github.com/MyoInteract/MyoInteract/pull/3)

GitHub listed these as open during this review. Their descriptions were checked to avoid proposing duplicate work; their patches were not independently reviewed here. In particular, fixing auto-reset is complementary to #1 because correct reset information must survive for its episode-local noise state to work as intended.

Recommended delivery sequence: establish a working CI job alongside the first correctness fixes, then land proposals 1-4 with focused regression tests. Follow with evaluation/provenance and configuration fixes. Validate affected reference tasks again before treating the corrected implementation as the new comparison baseline. Keep each PR scoped to its stated behavioral contract.
