# Configurations and output locations

The interface reads the three bundled example presets from the installed
`myosuite/data/gradio_configs` directory. It saves personal configurations under
the operating system's user data directory (`myointeract/gradio_configs`), and
training writes to the matching `myointeract/logs` directory. These locations do
not change when a notebook starts from a different working directory.

Set `MYOINTERACT_DATA_DIR` to move both personal directories. Set
`MYOINTERACT_CONFIG_DIR` or `MYOINTERACT_LOG_DIR` to override one directory.
For existing source checkouts, set `MYOINTERACT_CONFIG_DIR` to the checkout's
`gradio_configs` directory and `MYOINTERACT_LOG_DIR` to its `logs` directory to
continue using previous personal files. Bundled examples remain available.

The `ar` example preserves its reference to the source checkout's pretrained
`tracked_checkpoints/universal/ar/checkpoints/20019200` checkpoint. Wheels omit
these large pretrained files; use the source checkout for that checkpoint, or
choose `None` in the checkpoint selector to train a new policy. The mobile
typing and public display examples train from scratch.

New presets use schema version 1: a JSON object with `schema_version: 1` and a
`parameters` object. Each parameter has a stable name such as `rl.num_envs`,
`biomechanics.ctrl_dt`, or `spheres.0.x_range`. Object order has no meaning.
The original 178-value arrays remain readable and are migrated in memory;
saving a new copy writes the named-field format. The three bundled examples
preserve all values from their legacy counterparts. Unknown versions, missing
fields, and invalid field types produce an error instead of shifting controls.

Configuration names can include spaces and dots, such as `study.v2`. Names
cannot contain path separators or reserved filename characters, and saving
never replaces an existing preset.

Render Environment displays the current target seed without changing it.
Resample Targets chooses the next seed, renders it, and stores that same seed
for saving and training. The seed controls the initial target layout; training
continues to sample targets according to each task's reset behavior.
