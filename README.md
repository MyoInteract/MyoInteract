# MyoInteract (Beta)

This repository contains a beta version of the research code for MyoInteract.

This project is built on top of a fork of myosuite. Refer to [README_MYOSUITE.md](README_MYOSUITE.md) for that project's readme.

## Environment setup

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
