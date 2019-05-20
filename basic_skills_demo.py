#!/usr/bin/env python3

import sys, os

dir_name = sys.argv[1]

for filename in os.listdir(dir_name):
	if filename.startswith('.'):
		continue
	if not filename.endswith('.txt'):
		continue
	filepath = os.path.join(dir_name, filename)
	with open(filepath, 'r') as open_file:
		for line in open_file.readlines():
			print(line)
			break