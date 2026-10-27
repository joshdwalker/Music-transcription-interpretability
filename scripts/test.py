from pathlib import Path

greeting = 'Hello, World!'

Path('outputs/test.txt').write_text(greeting)