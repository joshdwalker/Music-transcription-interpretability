from pathlib import Path

import torch
from muscriptor import TranscriptionModel

text_file = '/depot/yunglu/data/transcription/slakh2100.txt'
text_paths = []

with open(text_file) as f:
    for line in f:
        text_paths.append(line.strip())

# Place each .pt file in activations dir from this repo
output_dir = 'activations'
# Ensure the subdirectory (already or not) exists in this repo
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Requires accepting Muscriptor's model license on Hugging Face.
transcription_model = TranscriptionModel.load_model("large", device='cuda')
model = transcription_model._model
# Print to ensure cuda worked
print("CUDA available:", torch.cuda.is_available())
print("Model device:", transcription_model._device)
print("GPU:", torch.cuda.get_device_name(0))

stored_activations: list[torch.Tensor] = []


def hook_fn(_module, _inputs, output: torch.Tensor) -> None:
    if not isinstance(output, torch.Tensor):
        raise TypeError(
            f"Expected tensor output from out_norm, got {type(output).__name__}"
        )
    stored_activations.append(
        output.detach()
        .to(device="cpu", dtype=torch.float32)
        .reshape(-1, output.shape[-1])
    )


# out_norm is the final normalization before the LM projection. Its output
# excludes prepended audio-conditioning positions and has shape [batch, tokens, hidden].
handle = model.out_norm.register_forward_hook(hook_fn)
for text_path in text_paths:
    sample_path = Path(text_path)
    # text_path is of the form /depot/.../(train/test/validation)/Track_____/Track_____.wav
    output_path = Path(output_dir + text_path[text_path.rindex('/') : text_path.rindex('.')] + '.pt')
    stored_activations.clear()

    try:
        with torch.inference_mode():
            for _ in transcription_model.transcribe(sample_path):
                pass
    finally:
        handle.remove()

    if not stored_activations:
        raise RuntimeError(f"No activations were captured while transcribing {sample_path}")

    activations = torch.cat(stored_activations, dim=0)
    torch.save(
        {
            "activations": activations,
            "audio_path": str(sample_path),
            "layer": "out_norm",
        },
        output_path,
    )
    print(f"Saved {activations.shape[0]} token activations to {output_path}")
