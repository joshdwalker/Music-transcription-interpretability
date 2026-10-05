from pathlib import Path
from muscriptor import TranscriptionModel

model = TranscriptionModel.load_model("small") # Change to large for GPUs

sample = 'jingle_bells_guitar_easy'

# Get a MIDI file directly:
Path('outputs/' + sample + '.mid').write_bytes(model.transcribe_and_postprocess('samples/' + sample + '.mp3')[0])

# Listen to the output in a web midi editor like signal