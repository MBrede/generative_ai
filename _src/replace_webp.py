import subprocess
import os
import re

files = [file for file in os.listdir('content/')
         if 'qmd' in file]
pattern = re.compile(r"(?<=\()\.\./.+?.webp")
for file in files:
    with open(os.path.join('content', file), 'r') as f:
        content = f.read()
    matches = pattern.findall(content)
    if matches:
        for match in matches:
            png_path = match[:-4] + 'png'
            content = re.sub(match, png_path, content)
            subprocess.call(['dwebp', match, '-o', png_path])
        with open(os.path.join('content', file), 'w') as f:
            f.write(content)