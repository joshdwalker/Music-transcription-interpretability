import torch
from muscriptor import TranscriptionModel
from transformers import AutoModelForCausalLM

# Load the model weights (Requires accepting the CC BY-NC 4.0 license on HF)
model = TranscriptionModel.load_model('small')._model

#delete
print(model.transformer)

# Storage dictionary for activations
stored_activations = {}

def hook_fn(module, input, output):
    # Depending on the architecture, 'output' might be a tuple. 
    # The first element is usually the hidden states tensor: [batch, seq_len, hidden_dim]
    if isinstance(output, tuple):
        hidden_states = output[0]
    else:
        hidden_states = output
        
    stored_activations["final_layer"] = hidden_states.detach().cpu()

# Find the final transformer block/layer before the LM head
# Replace 'model.model.layers[-1]' with the exact path from your print(model) call
final_layer = model.out_norm
handle = final_layer.register_forward_hook(hook_fn)
