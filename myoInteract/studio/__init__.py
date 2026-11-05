"""MyoInteract Studio: the guided web app (``static/index.html``) and its backend (``myo_studio serve``).

The backend wraps the existing pieces: presets in the Gradio format, ``myo_train`` for training
(one subprocess per run), and :mod:`myoInteract.studio.worker` for previews and evaluations.
"""
