"""Compatibility test entry point for the project's actual checkpoint path."""

from myosuite.tests.test_training_custom_ppo import (
    test_project_checkpoint_roundtrip_preserves_deterministic_policy,
)
