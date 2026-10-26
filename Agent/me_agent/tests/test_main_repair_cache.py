"""Main-entry cache routing and experiment isolation, using synthetic files only."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import extract


class MainRepairCacheTests(unittest.TestCase):
    def test_source_option_routes_to_cache_without_full_run(self):
        args = SimpleNamespace()
        with patch.dict(os.environ, {'REPAIR_UPSTREAM_ROOT': 'source'}), \
             patch.object(extract, '_repair_cache_follow_run') as follow:
            extract.single_gpu_run_command(args)
        follow.assert_called_once_with(extract, args)

    def test_cache_mode_cannot_silently_run_fresh_upstream(self):
        args = SimpleNamespace(repair_from_cache=False, fresh=True)
        with patch.dict(os.environ, {'REPAIR_UPSTREAM_ROOT': 'source'}):
            with self.assertRaisesRegex(ValueError, 'Upstream reuse requires'):
                extract.extract_command(args)

    def test_main_batch_runs_one_main_script_child_with_frozen_inputs(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            source, output = root/'source', root/'main_repair'
            paper = '1000.00001'
            start = '2026-01-01T00:00:00+00:00'
            monitor = source/'error_analysis/producer/monitor.json'
            extract._repair_cache_atomic_json(monitor, {'started_at': start,
                'status': 'complete', 'papers': [paper], 'model': 'gemma', 'gpu_id': '0'})
            paper_list = root/'papers.txt'
            paper_list.write_text(paper+' baseline30 baseline\n')
            args = SimpleNamespace(python='python3', gpu_id='0', model='gemma', repair_model='gemma',
                paper_list=paper_list, split='all', max_papers=30, start_at='', batch_id='main_test')
            batch = output/'error_analysis/main_test'
            def launch(command, env, **kwargs):
                self.assertEqual(command, ['bash', 'run.sh'])
                self.assertEqual(env['REPAIR_FROM_CACHE'], '1')
                self.assertEqual(env['FRESH_RUN'], '0')
                self.assertEqual(env['REPAIR_SOURCE_STARTED_AT'], start)
                self.assertNotIn('ME_EXTRACT_MODULE', env)
                self.assertEqual(env['REPAIR_UPSTREAM_ROOT'], str(source))
                extract._repair_cache_atomic_json(output/paper/'pan_agent_risk_refinement.json', {
                    'agent_trace_version': extract.PAN_REPAIR_VERSION, 'upstream_cache_reused': True,
                    'summary': {'infrastructure_stopped_task_count': 0}})
                (batch/'paper_status.jsonl').write_text(json.dumps({'paper_id': paper, 'status': 'success'})+'\n')
                return SimpleNamespace(poll=lambda: 0, returncode=0)
            with ExitStack() as stack:
                stack.enter_context(patch.dict(os.environ, {'REPAIR_UPSTREAM_ROOT': str(source)}, clear=True))
                stack.enter_context(patch.object(extract, 'OUTPUT_ROOT', root))
                stack.enter_context(patch.object(extract, 'DAY_NAME', 'main_repair'))
                stack.enter_context(patch.object(extract, '_single_gpu_gpu_processes', return_value=[]))
                stack.enter_context(patch.object(extract, '_single_gpu_gpu_stats', return_value={}))
                stack.enter_context(patch.object(extract.socket.socket, 'connect_ex', return_value=1))
                stack.enter_context(patch.object(extract.subprocess, 'run', return_value=SimpleNamespace(stdout='test')))
                launches = stack.enter_context(patch.object(extract.subprocess, 'Popen', side_effect=launch))
                stack.enter_context(patch('builtins.print'))
                extract.single_gpu_run_command(args)
            launches.assert_called_once()
            self.assertEqual(extract._repair_cache_read_json(batch/'monitor.json')['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
