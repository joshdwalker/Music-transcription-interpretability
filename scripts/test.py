from pathlib import Path

greeting = 'Hello, World!'

Path('activations').mkdir(parents=True, exist_ok=True)
Path('activations/test.txt').write_text(greeting)