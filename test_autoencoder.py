from pathlib import Path

import torch
from muscriptor import TranscriptionModel


SAMPLE_AUDIO = Path("samples/jingle_bells_guitar_easy.mp3")
OUTPUT_PATH = Path("outputs") / f"{SAMPLE_AUDIO.stem}_activations.pt"

# Requires accepting Muscriptor's model license on Hugging Face.
transcription_model = TranscriptionModel.load_model("small")
model = transcription_model._model

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
try:
    with torch.inference_mode():
        for _ in transcription_model.transcribe(SAMPLE_AUDIO):
            pass
finally:
    handle.remove()

if not stored_activations:
    raise RuntimeError(f"No activations were captured while transcribing {SAMPLE_AUDIO}")

activations = torch.cat(stored_activations, dim=0)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "activations": activations,
        "audio_path": str(SAMPLE_AUDIO),
        "layer": "out_norm",
    },
    OUTPUT_PATH,
)
print(f"Saved {activations.shape[0]} token activations to {OUTPUT_PATH}")
