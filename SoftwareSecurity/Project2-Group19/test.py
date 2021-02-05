from pathlib import Path
from unittest import TestCase

import main

SAMPLE_PATH = Path('examples') / 'proj-slices'
VULN_NAME = 'Command Injection'


class BasicFlowTest(TestCase):

    def test_flow_a(self):
        program_path = SAMPLE_PATH / '1a-basic-flow.json'
        patterns_path = SAMPLE_PATH / '1a-basic-flow-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["c"],
            "sink": ["e"],
            "sanitizer": []
        }]
        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_flow_b(self):
        program_path = SAMPLE_PATH / '1b-basic-flow.json'
        patterns_path = SAMPLE_PATH / '1b-basic-flow-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["e"],
            "sanitizer": []
        }]
        self.assertListEqual(main.main(program_path, patterns_path), expected)


class ExpressionTest(TestCase):
    def test_binary_ops(self):
        program_path = SAMPLE_PATH / '2-expr-binary-ops.json'
        patterns_path = SAMPLE_PATH / '2-expr-binary-ops-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["e"],
            "sanitizer": []
        }]
        self.assertListEqual(main.main(program_path, patterns_path), expected)


class FunctionCallTest(TestCase):
    def test_a(self):
        program_path = SAMPLE_PATH / '3a-expr-func-calls.json'
        patterns_path = SAMPLE_PATH / '3a-expr-func-calls-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["d"],
            "sanitizer": []
        }]
        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_a_sanitizer(self):
        program_path = SAMPLE_PATH / '3a-expr-func-calls.json'
        patterns_path = SAMPLE_PATH / '3a-expr-func-calls-pat2.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sanitizer": ["d"],
            "sink": ["e"]
        }]
        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_b(self):
        program_path = SAMPLE_PATH / '3b-expr-func-calls.json'
        patterns_path = SAMPLE_PATH / '3b-expr-func-calls-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["a"],
            "sanitizer": ["t"],
            "sink": ["z"]
        }]
        self.assertListEqual(main.main(program_path, patterns_path), expected)


# Might not get sanitized based on the condition, not sure how to handle
class ConditionBranching(TestCase):
    def test_a(self):
        program_path = SAMPLE_PATH / '4a-conds-branching.json'
        patterns_path = SAMPLE_PATH / '4a-conds-branching-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["e"],
            "sanitizer": []  # TODO: check - I think because of the second if statement there is also a flow
                             #  from source to sink without touching the sanitizer f
        }, {
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["e"],
            "sanitizer": ["f"]
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_a_not_sanitized(self):
        program_path = SAMPLE_PATH / '4a-conds-branching.json'
        patterns_path = SAMPLE_PATH / '4a-conds-branching-pat2.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sanitizer": [],  # TODO: check - I think [a] can only be tainted if it enters if(),
                              #  otherwise a is not tainted as only b is a source (b is not present in else{}
            "sink": ["e"]
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_conds_branching_b(self):
        program_path = SAMPLE_PATH / '4b-conds-branching.json'
        patterns_path = SAMPLE_PATH / '4b-conds-branching-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["c"],
            "sink": ["e"],
            "sanitizer": []
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)


class LoopUnfolding(TestCase):
    def test_a(self):
        program_path = SAMPLE_PATH / '5a-loops-unfolding.json'
        patterns_path = SAMPLE_PATH / '5a-loops-unfolding-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["d"],
            "sink": ["h"],
            "sanitizer": ["f"]
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_a_vuln(self):
        program_path = SAMPLE_PATH / '5a-loops-unfolding.json'
        patterns_path = SAMPLE_PATH / '5a-vuln-patterns.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sanitizer": [],
            "sink": ["d"]
        },
            {"vulnerability": VULN_NAME,
             "source": ["b"],
             "sanitizer": [],
             "sink": ["h"]
             }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_b(self):
        program_path = SAMPLE_PATH / '5b-loops-unfolding.json'
        patterns_path = SAMPLE_PATH / '5b-loops-unfolding-pat.json'

        expected = []  # TODO: check - I think as a is changed only with s(a, 1) it doesn't affect c => doesn't reach sink

        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_c(self):
        program_path = SAMPLE_PATH / '5c-loops-unfolding.json'
        patterns_path = SAMPLE_PATH / '5c-loops-unfolding-pat.json'

        expected = []  # TODO: check - maybe change the -path.json file

        self.assertListEqual(main.main(program_path, patterns_path), expected)


class Sanitization(TestCase):
    def test_sanitization_a(self):
        program_path = SAMPLE_PATH / '6a-sanitization.json'
        patterns_path = SAMPLE_PATH / '6a-sanitization-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["z"],
            "sanitizer": []
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_sanitization_b(self):
        program_path = SAMPLE_PATH / '6b-sanitization.json'
        patterns_path = SAMPLE_PATH / '6b-sanitization-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["e"],
            "sanitizer": ["c"]
        }, {
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["e"],
            "sanitizer": []
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_sanitization_b_ok(self):
        program_path = SAMPLE_PATH / '6b-sanitization.json'
        patterns_path = SAMPLE_PATH / '6b-sanitization-pat2.json'

        expected = []

        self.assertListEqual(main.main(program_path, patterns_path), expected)

    def test_sanitization_b_unsanitized(self):
        program_path = SAMPLE_PATH / '6b-sanitization.json'
        patterns_path = SAMPLE_PATH / '6b-sanitization-pat3.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["c"],
            "sanitizer": []
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)


class ConditionImplicit(TestCase):
    def test(self):
        program_path = SAMPLE_PATH / '7-conds-implicit.json'
        patterns_path = SAMPLE_PATH / '7-conds-implicit-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source": ["b"],
            "sink": ["e"],
            "sanitizer": []
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)

class LoopImplicit(TestCase):
    def test(self):
        program_path = SAMPLE_PATH / '8-loops-implicit.json'
        patterns_path = SAMPLE_PATH / '8-loops-implicit-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source":["b"],
            "sink":["w"],
            "sanitizer":[]
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)

class RegionGuard(TestCase):
    def test(self):
        program_path = SAMPLE_PATH / '9-regions-guards.json'
        patterns_path = SAMPLE_PATH / '9-regions-guards-pat.json'

        expected = [{
            "vulnerability": VULN_NAME,
            "source":["b"],
            "sink":["z"],
            "sanitizer":[]
        }]

        self.assertListEqual(main.main(program_path, patterns_path), expected)
