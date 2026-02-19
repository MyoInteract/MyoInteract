# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO multimodal vision networks."""

from typing import Any, Callable, Mapping, Sequence, Tuple

from brax.training import distribution
from brax.training.networks import FeedForwardNetwork, make_policy_network, make_value_network
from brax.training import types
from brax.training.types import PRNGKey
import flax
from flax import linen
import jax.numpy as jnp
import functools
from dataclasses import dataclass
import jax

ModuleDef = Any
ActivationFn = Callable[[jnp.ndarray], jnp.ndarray]
Initializer = Callable[..., Any]

@flax.struct.dataclass
class PPONetworksUnifiedExtractor:
  feature_extractor: FeedForwardNetwork
  policy_network: FeedForwardNetwork
  value_network: FeedForwardNetwork
  parametric_action_distribution: distribution.ParametricDistribution

@dataclass
class CNNLayer:
  features: int
  kernel_size: Tuple[int, int]
  stride: Tuple[int, int]
  padding: Tuple[int, int]
  use_bias: bool = True


class MultimodalFeatureExtractor(linen.Module):
  """Multimodal feature extractor with vision and proprioception branches."""
  
  vision_output_size: int = 256
  proprioception_output_size: int = 128
  proprioception_obs_key: str = 'proprioception'
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01)
  cnn_layers: Sequence[CNNLayer] = (
    CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
  )
  normalise_pixels: bool = True
  
  @linen.compact
  def __call__(self, data: dict):
    # Vision branch
    vision_features = []
    pixels_hidden = {k: v for k, v in data.items() if k.startswith('pixels/')}

    assert len(pixels_hidden) >= 1, "At least one vision input is required"

    if self.normalise_pixels:
      # Calculates shared statistics over an entire 2D image.
      image_layernorm = functools.partial(
          linen.LayerNorm,
          use_bias=False,
          use_scale=False,
          reduction_axes=(-1, -2),
      )

      def ln_per_chan(v: jax.Array):
        normalised = [
            image_layernorm()(v[..., chan]) for chan in range(v.shape[-1])
        ]
        return jnp.stack(normalised, axis=-1)

      pixels_hidden = jax.tree.map(ln_per_chan, pixels_hidden)
    
    for key in pixels_hidden:
      # CNN layers for vision
      hidden = pixels_hidden[key]

      for layer in self.cnn_layers:
        hidden = linen.Conv(
          features=layer.features,
          kernel_size=layer.kernel_size,
          strides=layer.stride,
          padding=layer.padding,
          use_bias=layer.use_bias,
        )(hidden)
        hidden = self.activation(hidden)
      
      # Flatten
      spatial_dims = hidden.ndim - 3  # Number of leading dimensions to preserve
      hidden = jnp.reshape(hidden, hidden.shape[:spatial_dims] + (-1,))
      vision_features.append(hidden)
      
    # Concatenate all vision features and pass through linear layer
    vision_concat = jnp.concatenate(vision_features, axis=-1)
    vision_out = linen.Dense(
        features=self.vision_output_size,
        use_bias=True,
    )(vision_concat)
    vision_out = self.activation(vision_out)
    
    
    # Proprioception branch
    assert self.proprioception_obs_key in data, "Proprioception input is required"
    
    proprioception_input = data[self.proprioception_obs_key]
    proprioception_out = linen.Dense(
        features=self.proprioception_output_size,
        use_bias=True,
    )(proprioception_input)
    proprioception_out = self.activation(proprioception_out)
    
    # Concatenate vision and proprioception features
    return jnp.concatenate([vision_out, proprioception_out], axis=-1)

def make_unified_feature_extractor(
    observation_size: Mapping[str, Tuple[int, ...]],
    preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
    vision_output_size: int = 256,
    proprioception_output_size: int = 128,
    proprioception_obs_key: str = 'proprioception',
    activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01),
    cnn_layers: Sequence[CNNLayer] = (
      CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
      CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    ),
    normalise_pixels: bool = True,
 ) -> FeedForwardNetwork:
  """Make unified feature extractor."""
  feature_extractor = MultimodalFeatureExtractor(
    vision_output_size=vision_output_size,
    proprioception_output_size=proprioception_output_size,
    proprioception_obs_key=proprioception_obs_key,
    activation=activation,
    cnn_layers=cnn_layers,
    normalise_pixels=normalise_pixels,
  )

  def apply(processor_params, extractor_params, obs):
    #TODO: fix this properly
    proprioception_obs = preprocess_observations_fn(
      obs["proprioception"], processor_params
    )
    obs = {**obs, "proprioception": proprioception_obs}

    features = feature_extractor.apply(extractor_params, obs)

    return features

  dummy_obs = {
      key: jnp.zeros((1,) + shape) for key, shape in observation_size.items()
  }
  return FeedForwardNetwork(
      init=lambda key: feature_extractor.init(key, dummy_obs), apply=apply
  )

def make_ppo_networks_unified_extractor(
  observation_size: Mapping[str, Tuple[int, ...]],
  action_size: int,
  preprocess_observations_fn: types.PreprocessObservationFn = types.identity_observation_preprocessor,
  policy_hidden_layer_sizes: Sequence[int] = (256, 256),
  value_hidden_layer_sizes: Sequence[int] = (256, 256),
  activation: ActivationFn = functools.partial(linen.leaky_relu, negative_slope=0.01),
  normalise_pixels: bool = True,
  vision_output_size: int = 256,
  proprioception_output_size: int = 128,
  proprioception_obs_key: str = 'proprioception',
  cnn_layers: Sequence[CNNLayer] = (
    CNNLayer(features=8, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
    CNNLayer(features=32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
  )
) -> PPONetworksUnifiedExtractor:
  """Make PPO networks with unified feature extractor."""

  parametric_action_distribution = distribution.NormalTanhDistribution(
      event_size=action_size
  )

  feature_extractor = make_unified_feature_extractor(
    observation_size=observation_size,
    preprocess_observations_fn=preprocess_observations_fn,
    vision_output_size=vision_output_size,
    proprioception_output_size=proprioception_output_size,
    proprioception_obs_key=proprioception_obs_key,
    activation=activation,
    cnn_layers=cnn_layers,
    normalise_pixels=normalise_pixels,
  )

  feature_extractor_output_size = vision_output_size + proprioception_output_size
  
  policy_network = make_policy_network(
      parametric_action_distribution.param_size,
      feature_extractor_output_size,
      # Already preprocessed in the feature extractor so no need to preprocess again
      preprocess_observations_fn=types.identity_observation_preprocessor,
      hidden_layer_sizes=policy_hidden_layer_sizes,
      activation=activation,
  )
  value_network = make_value_network(
      feature_extractor_output_size,
      # Already preprocessed in the feature extractor so no need to preprocess again
      preprocess_observations_fn=types.identity_observation_preprocessor,
      hidden_layer_sizes=value_hidden_layer_sizes,
      activation=activation,
  )

  return PPONetworksUnifiedExtractor(
    feature_extractor=feature_extractor,
    policy_network=policy_network,
    value_network=value_network,
    parametric_action_distribution=parametric_action_distribution,
  )


def make_inference_fn_unified_extractor(ppo_network: PPONetworksUnifiedExtractor):
  """Creates params and inference function for the PPO agent."""
  feature_extractor = ppo_network.feature_extractor
  policy_network = ppo_network.policy_network
  parametric_action_distribution = ppo_network.parametric_action_distribution
  
  def make_policy(
      params: types.Params, deterministic: bool = False
  ) -> types.Policy:
    processor_params = params[0]
    extractor_params = params[1]
    policy_params = params[2]
    
    def policy(
        observations: types.Observation, key_sample: PRNGKey
    ) -> Tuple[types.Action, types.Extra]:
      extractor_outputs = feature_extractor.apply(processor_params, extractor_params, observations)
      # Policy network expects some kind of normalizer params, even though the network uses an identity preprocessor
      logits = policy_network.apply(None, policy_params, extractor_outputs)

      if deterministic:
        return parametric_action_distribution.mode(logits), {}

      raw_actions = parametric_action_distribution.sample_no_postprocessing(
          logits, key_sample
      )
      log_prob = parametric_action_distribution.log_prob(logits, raw_actions)
      postprocessed_actions = parametric_action_distribution.postprocess(
          raw_actions
      )
      return postprocessed_actions, {
          'log_prob': log_prob,
          'raw_action': raw_actions,
      }

    return policy

  return make_policy
    
    
  