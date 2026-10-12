from copy import deepcopy
import json
from pathlib import Path
import unittest
from mini_agentx.agents.brainstorm import citation_examples
from mini_agentx.agents.brainstorm_contracts import validate_batch


class CitationGuidanceTest(unittest.TestCase):
    def test_examples_use_real_direct_fields_and_preserve_types(self):
        evidence = {'evidence_id': 'tool-2', 'facts': {'validation_metrics': {'ndcg@10': .04}, 'training_seed': 42}}
        examples = citation_examples(evidence)
        self.assertEqual(examples[0], {'evidence_id': 'tool-2', 'field': 'validation_metrics', 'value': {'ndcg@10': .04}})
        self.assertEqual(examples[1]['value'], 42)
        self.assertTrue(all(ref['field'] in evidence['facts'] for ref in examples))

    def test_unknown_field_explains_available_names_not_none(self):
        example = Path(__file__).resolve().parents[1] / 'examples/brainstorm'
        context = json.loads((example / 'context.json').read_text())
        batch = json.loads((example / 'proposals.json').read_text())
        batch['proposals'][0]['evidence_refs'][0]['field'] = 'facts'
        with self.assertRaisesRegex(ValueError, '证据字段不存在.*可选'):
            validate_batch(batch, context)
