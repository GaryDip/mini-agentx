from pathlib import Path
import tempfile
import unittest

from mini_agentx.tools.brainstorm import BrainstormTools
from mini_agentx.llm.function_calling import schemas


class DiscoveryTest(unittest.TestCase):
    def test_browse_search_and_read_discovered_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'src/model').mkdir(parents=True)
            file = root / 'src/model/new_strategy.py'
            file.write_text('def sample():\n    return "hard"\n')
            tools = BrainstormTools(root, root / 'runs', {'parent_run_id': 'baseline-0123456789ab'})
            _, listing = tools.execute('list_directory', {'path': 'src/model'})
            self.assertEqual(listing['entries'][0]['path'], 'src/model/new_strategy.py')
            _, found = tools.execute('search_code', {'path': 'src', 'query': 'hard'})
            self.assertEqual(found['matches'][0]['line'], 2)
            _, result = tools.execute('read_file', {'path': found['matches'][0]['path']})
            self.assertIn('hard', result['content'])
            definitions = schemas(tools.SPECS, 'propose')
            read = next(item for item in definitions if item['function']['name'] == 'read_file')
            self.assertNotIn('enum', read['function']['parameters']['properties']['path'])

    def test_private_paths_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'src').mkdir()
            (root / '.env').write_text('private')
            (root / 'src/alias.py').symlink_to(root / '.env')
            tools = BrainstormTools(root, root / 'runs', {'parent_run_id': 'baseline-0123456789ab'})
            for path in ('.env', 'src/../.env', 'runs/report.json', str(root / '.env'), 'src/alias.py'):
                with self.assertRaises(ValueError): tools.execute('read_file', {'path': path})
            _, found = tools.execute('search_code', {'path': 'src', 'query': 'private'})
            self.assertEqual(found['matches'], [])

    def test_search_results_bounded(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'src').mkdir()
            (root / 'src/a.py').write_text('match\n' * 70)
            tools = BrainstormTools(root, root / 'runs', {'parent_run_id': 'baseline-0123456789ab'})
            _, result = tools.execute('search_code', {'path': 'src', 'query': 'match'})
            self.assertEqual(len(result['matches']), 50)
            self.assertTrue(result['truncated'])
