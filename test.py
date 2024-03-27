import torch
from torchviz import make_dot

# Define the model architecture string
model_architecture = """
MultiInputActorCriticPolicy(
  (features_extractor): CustomCombinedExtractor(
    (extractors): ModuleDict(
      (image): Sequential(
        (0): Conv2d(5, 32, kernel_size=(1, 1), stride=(4, 4))
        (1): ReLU()
        (2): Flatten(start_dim=1, end_dim=-1)
      )
      (vector): Sequential(
        (0): Linear(in_features=8, out_features=64, bias=True)
        (1): ReLU()
        (2): Linear(in_features=64, out_features=32, bias=True)
        (3): ReLU()
      )
    )
  )
  (pi_features_extractor): CustomCombinedExtractor(
    (extractors): ModuleDict(
      (image): Sequential(
        (0): Conv2d(5, 32, kernel_size=(1, 1), stride=(4, 4))
        (1): ReLU()
        (2): Flatten(start_dim=1, end_dim=-1)
      )
      (vector): Sequential(
        (0): Linear(in_features=8, out_features=64, bias=True)
        (1): ReLU()
        (2): Linear(in_features=64, out_features=32, bias=True)
        (3): ReLU()
      )
    )
  )
  (vf_features_extractor): CustomCombinedExtractor(
    (extractors): ModuleDict(
      (image): Sequential(
        (0): Conv2d(5, 32, kernel_size=(1, 1), stride=(4, 4))
        (1): ReLU()
        (2): Flatten(start_dim=1, end_dim=-1)
      )
      (vector): Sequential(
        (0): Linear(in_features=8, out_features=64, bias=True)
        (1): ReLU()
        (2): Linear(in_features=64, out_features=32, bias=True)
        (3): ReLU()
      )
    )
  )
  (mlp_extractor): MlpExtractor(
    (policy_net): Sequential(
      (0): Linear(in_features=96, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
    (value_net): Sequential(
      (0): Linear(in_features=96, out_features=64, bias=True)
      (1): Tanh()
      (2): Linear(in_features=64, out_features=64, bias=True)
      (3): Tanh()
    )
  )
  (action_net): Linear(in_features=64, out_features=4, bias=True)
  (value_net): Linear(in_features=64, out_features=1, bias=True)
)
"""

# Create a dummy input to trace the model
dummy_input = torch.rand(1, 5)  # Change the input shape if needed

# Create a graph from the model architecture string
graph = make_dot(model_architecture, params=dict(model.named_parameters()), input_names=["input"])

# Save the graph to a file or display it
graph.render("model_architecture", format="png")  # Change the file format if needed
