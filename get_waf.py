#!/usr/bin/env python
import os
from urllib.request import urlopen


version = '2.0.27'
site = f'https://waf.io/waf-{version}'
exe = 'waf'


with urlopen(site) as data:
    with open(exe, 'w+b') as waf:
        waf.write(data.read())
    os.chmod(exe, 0o755)
