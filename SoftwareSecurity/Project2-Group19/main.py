import json
from pprint import pprint

import fire

import models
from models import Program, Pattern


def to_model(d):
    if isinstance(d, (str, int, float)) or d is None:
        return d
    if isinstance(d, dict):
        klass = getattr(models, d['type'])
        args = {k: to_model(v) for k, v in d.items()}
        return klass(**args)
    elif isinstance(d, list):
        return [to_model(v) for v in d]
    else:
        raise ValueError(f'Unsupported type: {type(d)}')


def main(program_path: str, patterns_path: str):
    with open(program_path) as f:
        program_json = json.loads(f.read().replace('async', 'is_async'))
    program: Program = to_model(program_json)
    # print('Program:')
    # pprint(program_json)

    with open(patterns_path) as f:  patterns = json.load(f, object_hook=lambda d: Pattern(**d))
    # print('Patterns:')
    # pprint(patterns)

    state = models.State(patterns)
    program.execute(state)
    # print('Before Vulnerabilities:')
    # pprint(state.vulnerabilities)
    vulnerabilities = [v.__dict__ for v in state.vulnerabilities]
    vulnerabilities = [{k: list(v) if isinstance(v, tuple) else v
                        for k, v in vuln.items()}
                       for vuln in vulnerabilities]
    # print('After Vulnerabilities:')
    # pprint(vulnerabilities)
    return vulnerabilities


if __name__ == '__main__':
    fire.Fire(main)
