"""Offline orchestration tests. No model, CUDA, network or annotation contents used."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import extract
import extract as runner


class SingleGpuTests(unittest.TestCase):
    def test_seed_contract_logging_reaches_model_request(self):
        class RequestReached(BaseException):
            pass

        client = Mock()
        client.chat.completions.create.side_effect = RequestReached
        state = SimpleNamespace(paper_id='fictional', trajectory=[])
        with tempfile.TemporaryDirectory() as td:
            events = Path(td)/'events.jsonl'
            with patch.object(extract, 'build_release_baseline_seed_prompt', return_value='synthetic prompt'):
                with self.assertRaises(RequestReached):
                    extract.extract_baseline_style_pan_seed(
                        client, 'synthetic-model', {}, {}, 'synthetic paper',
                        state=state, event_log_path=events, max_tokens=6500)
            self.assertIn('global_seed_request_contract', events.read_text(encoding='utf-8'))
        client.chat.completions.create.assert_called_once()
        self.assertEqual(state.trajectory[-1]['target'], 'global_seed')
        self.assertEqual(state.trajectory[-1]['max_output_tokens'], 12288)

    def test_phase_routing_never_colocates_models(self):
        args = SimpleNamespace(python='python3', gpu_id='0', model='9b', repair_model='35b', batch_id='test')
        first = runner._single_gpu_phase_env({}, 'upstream', args, 'selected.txt')
        second = runner._single_gpu_phase_env({}, 'repair', args, 'selected.txt')
        self.assertEqual(first['REPAIR_MODEL'], '9b')
        self.assertEqual(first['UPSTREAM_ONLY'], '1')
        self.assertEqual(second['MODEL'], '9b')
        self.assertEqual(second['REPAIR_MODEL'], '35b')
        self.assertEqual(second['REPAIR_FROM_CACHE'], '1')
        self.assertEqual(second['FRESH_RUN'], '0')
        self.assertEqual(first['GPU_ID'], second['GPU_ID'])
        self.assertEqual(second['REPAIR_GPU_ID'], '')

    def test_selection_respects_split_start_limit_and_no_duplicates(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'papers.txt'
            p.write_text('# comment\n1000.00001 dev simple\n1000.00002 validation complex\n1000.00003 dev simple\n')
            self.assertEqual(runner._single_gpu_select_rows(p, 'all', 1, '1000.00002')[0][0], '1000.00002')
            self.assertEqual(len(runner._single_gpu_select_rows(p, 'dev', 0, '')), 2)
            with self.assertRaises(ValueError):
                runner._single_gpu_select_rows(p, 'dev', 0, 'missing')
            p.write_text('1000.00001 dev simple\n1000.00001 dev simple\n')
            with self.assertRaises(ValueError):
                runner._single_gpu_select_rows(p, 'all', 0, '')

    def test_monitor_excludes_old_calls_deduplicates_and_separates_models(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'1000.00001'; p.mkdir()
            def event(stamp, model, cid, total):
                return {'timestamp': stamp, 'type': 'llm_call', 'llm_call': {'model': model,
                    'call_id': cid, 'total_tokens': total, 'prompt_tokens': total, 'completion_tokens': 0,
                    'usage_available': True}}
            old = event('2020-01-01T00:00:00+00:00', '9b', 'old', 999)
            a = event('2026-01-01T00:00:01+00:00', '9b', 'a', 10)
            b = event('2026-01-01T00:00:02+00:00', '35b', 'b', 20)
            (p/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in [old, a, a, b])+'\n{"partial"')
            result = runner._single_gpu_collect_usage(Path(td), [p.name], '2026-01-01T00:00:00+00:00')
            self.assertEqual(result['by_model']['9b']['total_tokens'], 10)
            self.assertEqual(result['by_model']['35b']['total_tokens'], 20)
            self.assertEqual(result['per_paper'][p.name]['calls'], 2)

    def test_upstream_boundary_validates_cache_and_never_repairs(self):
        state = SimpleNamespace(paper_id='fictional', pan_seed_record={'samples': []})
        with patch.object(extract, 'save_checkpoint') as saved, \
             patch.object(extract, '_load_cached_pan_repair_inputs') as validated, \
             patch.object(extract, 'append_event') as logged, \
             patch.object(extract, 'run_pan_agent_risk_refinement') as repair:
            result, same = extract._finish_upstream(state, Path('x/checkpoint.json'), Path('events'))
        self.assertIs(state, same)
        self.assertEqual(result, state.pan_seed_record)
        self.assertIsNot(result, state.pan_seed_record)
        self.assertEqual(saved.call_count, 2)
        validated.assert_called_once()
        logged.assert_called_once()
        repair.assert_not_called()

    def test_bad_upstream_cache_does_not_emit_completion(self):
        state = SimpleNamespace(paper_id='fictional')
        with patch.object(extract, 'save_checkpoint'), \
             patch.object(extract, '_load_cached_pan_repair_inputs', side_effect=RuntimeError('incomplete detector')), \
             patch.object(extract, 'append_event') as logged:
            with self.assertRaises(RuntimeError):
                extract._finish_upstream(state, Path('x/checkpoint.json'), Path('events'))
            logged.assert_not_called()

    def run_mock_pipeline(self, fail_upstream=False, same_model=False):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root/'scripts').mkdir()
            paper = '1000.00001'
            (root/'papers.txt').write_text(paper+' evaluation simple\n')
            (root/'SOURCES/input_papers'/paper).mkdir(parents=True)
            (root/'SOURCES/input_papers'/paper/'paper.pdf').write_bytes(b'fixture')
            (root/'SOURCES/gold_data').mkdir(parents=True)
            (root/'SOURCES/gold_data'/f'{paper}.json').write_text('{}')
            old = root/'outputs/run'/paper; old.mkdir(parents=True)
            (old/'prediction.json').write_text('old results')
            phases = []
            def launch(command, env, **kwargs):
                phase = env['SINGLE_GPU_PHASE']; phases.append(phase)
                if phase == 'full':
                    self.assertEqual(env['MODEL'], env['REPAIR_MODEL'])
                    self.assertEqual(env['UPSTREAM_ONLY'], '0')
                    self.assertEqual(env['REPAIR_FROM_CACHE'], '0')
                    self.assertEqual(env['FRESH_RUN'], '1')
                status = 'extraction_failed' if phase == 'upstream' and fail_upstream else (
                    'upstream_success' if phase == 'upstream' else 'success')
                out = root/'outputs/run/error_analysis'/env['BATCH_RUN_ID']; out.mkdir(parents=True, exist_ok=True)
                (out/'paper_status.jsonl').write_text(json.dumps({'paper_id': paper, 'status': status})+'\n')
                return SimpleNamespace(poll=lambda: 0, returncode=0)
            sock = Mock(); sock.__enter__ = Mock(return_value=sock); sock.__exit__ = Mock(return_value=False)
            sock.connect_ex.return_value = 1
            argv = ['extract.py', 'single-gpu-run', '--python', 'python3', '--gpu-id', '0', '--model', '9b', '--repair-model', '35b',
                    '--paper-list', 'papers.txt', '--split', 'all', '--batch-id', 'test']
            previous = os.getcwd()
            if same_model:
                argv[argv.index('--model') + 1] = 'google/gemma-4-31B-it'
                argv[argv.index('--repair-model') + 1] = 'google/gemma-4-31B-it'
            try:
                with patch.object(runner, '__file__', str(root/'extract.py')), \
                     patch('sys.argv', argv), patch.dict(os.environ, {}, clear=True), \
                     patch.object(runner, '_single_gpu_gpu_processes', return_value=[]), \
                     patch.object(runner, '_single_gpu_gpu_memory_used', return_value=0), \
                     patch.object(runner, '_single_gpu_gpu_stats', return_value={}), \
                     patch.object(runner.socket, 'socket', return_value=sock), \
                     patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout='testcommit')), \
                     patch.object(runner.subprocess, 'Popen', side_effect=launch), \
                     patch.object(runner.signal, 'signal'), patch('builtins.print'):
                    if fail_upstream:
                        with self.assertRaisesRegex(RuntimeError, 'upstream incomplete'):
                            runner.main()
                    else:
                        runner.main()
                batch = root/'outputs/run/error_analysis/test'
                self.assertEqual((batch/'previous_outputs'/paper/'prediction.json').read_text(), 'old results')
                monitor = json.loads((batch/'monitor.json').read_text())
                self.assertEqual(monitor['status'], 'failed_or_interrupted' if fail_upstream else 'complete')
                self.assertEqual(phases, ['full'] if same_model else (['upstream'] if fail_upstream else ['upstream', 'repair']))
            finally:
                os.chdir(previous)

    def test_full_orchestration_two_phases_and_backup(self):
        self.run_mock_pipeline()

    def test_upstream_failure_never_launches_repair(self):
        self.run_mock_pipeline(fail_upstream=True)

    def test_same_model_runs_full_pipeline_once(self):
        self.run_mock_pipeline(same_model=True)

    def test_full_phase_preserves_isolated_output_and_port(self):
        args = SimpleNamespace(python='python3', gpu_id='0', model='gemma', repair_model='gemma', batch_id='test')
        env = runner._single_gpu_phase_env({'ME_OUTPUT_ROOT': 'outputs/experiments', 'ME_RUN_NAME': 'gemma_test', 'VLLM_PORT': '8002'}, 'full', args, 'selected.txt')
        self.assertEqual(env['ME_OUTPUT_ROOT'], 'outputs/experiments')
        self.assertEqual(env['ME_RUN_NAME'], 'gemma_test')
        self.assertEqual(env['VLLM_PORT'], '8002')
        self.assertEqual(env['BATCH_RUN_ID'], 'test')


if __name__ == '__main__':
    unittest.main()
