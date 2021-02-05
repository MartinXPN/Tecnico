import json
import os
import sys
from pathlib import Path

import main


# Disable
def blockprint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enableprint():
    sys.stdout = sys.__stdout__


test_ids = ['10', '14', '22', '31',]
if len(sys.argv) > 1:
    test_ids = sys.argv[1:]

for test_id in test_ids:
    pwd = Path.cwd()
    git_tests = pwd / 'examples' / 'git_tests'
    wrong_tests, succeeded_list = [], []

    blockprint()
    for folder_path in sorted(git_tests.glob('T' + test_id + '*')):
        test_number = folder_path.name.split('-')[1]
        folder_path = Path(folder_path)
        patterns_path = folder_path / 'patterns.json'
        program_path = folder_path / 'program.json'
        expected_program_output_path = folder_path / 'program.output.json'

        if not os.path.exists(expected_program_output_path):
            expected_program_output_path = folder_path / 'output.json'
        if not os.path.exists(program_path):
            program_path = folder_path / 'input.json'

        with open(expected_program_output_path) as f:
            expected_program_output_json = json.load(f)

        program_output = main.main(program_path, patterns_path)
        x, y = json.dumps(program_output, sort_keys=True), json.dumps(expected_program_output_json, sort_keys=True)
        if x == y:
            succeeded_list.append(test_number)
            print(f'Success in test {folder.name}')
        else:
            wrong_tests.append(test_number)
            enableprint()
            print(test_number)
            print('Prog:    ', x)
            print('Expected:', y)
            blockprint()

    enableprint()
    print('\n')
    print(f'Succeeded in {len(succeeded_list)} tests: \t\t {succeeded_list}')
    print(f'Wrong in {len(wrong_tests)} tests:   \t\t {wrong_tests}')
