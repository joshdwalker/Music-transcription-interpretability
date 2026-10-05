from pathlib import Path
from muscriptor import TranscriptionModel

# "large" resolves to hf://MuScriptor/muscriptor-large and downloads on first use.
model = TranscriptionModel.load_model("large")

# Get a MIDI file directly:
Path("out.mid").write_bytes(model.transcribe_to_midi("audio.wav"))

# Or stream note events as they are transcribed:
for event in model.transcribe("audio.wav"):
    print(event)  # NoteStartEvent / NoteEndEvent / ProgressEvent
