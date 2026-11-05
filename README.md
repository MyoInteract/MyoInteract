> **This branch is deprecated.** It holds the beta version of MyoInteract that accompanies the paper (DIS 2026).
> The maintained, stable version is on the [`main` branch](https://github.com/MyoInteract/MyoInteract), and the easiest
> way in is [MyoInteract Studio](https://myointeract.github.io/studio/) in your browser.

# MyoInteract (Beta)

This repository contains a beta version of the research code for MyoInteract.

This project is built on top of a fork of myosuite. Refer to [README_MYOSUITE.md](README_MYOSUITE.md) for that project's readme.

## Environment setup

Supported Python versions are 3.11 and 3.12. The CPU runtime works on Linux and
Windows; CUDA training requires a supported Linux JAX/CUDA installation.

For a locked development/test environment, install [uv](https://docs.astral.sh/uv/)
and run:

```bash
uv sync --extra test
uv run python scripts/prepare_test_assets.py
uv run pytest -q
```

The asset preparation command creates a sparse, pinned checkout of Playground's
import-time Menagerie dependency. MyoInteract uses its own included arm models.
The test suite covers MyoInteract; inherited MyoSuite asset-dependent suites can
still be invoked explicitly from their original files.

On a Linux CUDA machine, add `--extra cuda` to `uv sync`. Vision additionally
requires a compatible Madrona-MJX installation; its GPU rendering tests need
that runtime and are separate from the CPU renderer-contract tests.

The original conda setup remains available for Linux GPU environments:

Create a conda environment from `env.yml`:

```bash
conda env create -f env.yml
```

To use a different environment name:

```bash
conda env create -f env.yml -n your_env_name
```

Activate the environment (default name is `myointeract_env`):

```bash
conda activate myointeract_env
```

## Running the UI

Open **myointeract_ui.ipynb** and follow the instructions there to setup the interaction tasks and to run training. There are three existing pre-saved configurations which you can load from. Alternatively, you can also create your own interaction task!

Models and presets load independently of the notebook's working directory.
Personal presets and logs are stored in the operating system's user data
directory. See [configuration paths and preset migration](docs/presets.md).
The preview preserves its seed; use **Resample Targets** to choose a new layout.

Each training run records its effective configuration, source revision, package
versions, hardware/backend, seeds, and checkpoint restore semantics in
`run_manifest.json`. Policy loading with `rl.load_checkpoint_path` is a parameter
warm start. Single-host nonvision runs also save full snapshots under `training_state/`;
use `rl.resume_training_path` to restore optimizer state, training progress,
RNG and rollout state with the original total budget and configuration.
See [exact continuation and reproducibility](docs/reproducibility.md#exact-training-continuation).

## Contributors

<table>
<tr>
<td width="80" style="padding: 4px 8px 4px 0;">
<a href="https://github.com/ankit-bhattarai">
<img src="https://github.com/ankit-bhattarai.png" width="80" height="80" alt="Ankit Bhattarai" style="border-radius: 50%; object-fit: cover; vertical-align: middle;"/>
</a>
</td>
<td><strong><a href="https://github.com/ankit-bhattarai">Ankit Bhattarai</a></strong></td>
</tr>
<tr>
<td width="80" style="padding: 4px 8px 4px 0;">
<a href="https://github.com/Hannahselder">
<img src="https://github.com/Hannahselder.png" width="80" height="80" alt="Hannah Selder" style="border-radius: 50%; object-fit: cover; vertical-align: middle;"/>
</a>
</td>
<td><strong><a href="https://github.com/Hannahselder">Hannah Selder</a></strong></td>
</tr>
<tr>
<td width="80" style="padding: 4px 8px 4px 0;">
<a href="https://github.com/fl0fischer">
<img src="https://github.com/fl0fischer.png" width="80" height="80" alt="Florian Fischer" style="border-radius: 50%; object-fit: cover; vertical-align: middle;"/>
</a>
</td>
<td><strong><a href="https://github.com/fl0fischer">Florian Fischer</a></strong></td>
</tr>
</table>

## Citation

If you use this work, please cite the paper below:

```bibtex
@misc{bhattarai2026myointeractframeworkfastprototyping,
  title={MyoInteract: A Framework for Fast Prototyping of Biomechanical HCI Tasks using Reinforcement Learning},
  author={Ankit Bhattarai and Hannah Selder and Florian Fischer and Arthur Fleig and Per Ola Kristensson},
  year={2026},
  eprint={2602.15245},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2602.15245},
}
```

Paper: [arXiv:2602.15245](https://arxiv.org/abs/2602.15245)
