from pathlib import Path

greeting = 'Hello, World!'

Path('activations/test.txt').write_text(greeting)