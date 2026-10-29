import os
from pathlib import Path

def collect_text_paths(text_file='/depot/yunglu/data/transcription/slakh2100.txt', dir='activations'):
    output = []

    with open(text_file) as f:
        output = []

        for line in f:
            corresponding_pt_filename = line[line.rfind('/') + 1 : line.rfind('.')] + '.pt'

            # Skip files already present in dir (activations)
            if os.listdir(dir).__contains__(corresponding_pt_filename):
                continue

            output.append(line.strip())

        return output

text_paths = collect_text_paths(text_file='/depot/yunglu/data/transcription/slakh2100.txt', dir='activations')

# Print text_paths head and tail
print(f'head: {text_paths[ : 6]}')
print(f'tail: {text_paths[-6 : ]}')