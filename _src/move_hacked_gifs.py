import os
import re
import subprocess

files = [file for file in os.listdir('content/')
         if 'qmd' in file]
pattern = re.compile(r"(?<=include_gif\([\'\"]../)[^\.]+")
for file in files:
    with open(os.path.join('content', file), 'r') as f:
        content = f.read()
    matches = pattern.findall(content)
    if matches:
        for match in matches:
            subprocess.call(['cp', match + '.gif', 'docs/' + match + '.gif'])
            subprocess.call(['cp', match + '.png', 'docs/' + match + '.png'])