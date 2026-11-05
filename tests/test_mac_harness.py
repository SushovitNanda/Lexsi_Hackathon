import importlib.util
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys
from threading import Thread
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/mac'))
from run_cell import answer_script, digest, load_rows, one_prompt, resume_rows
from verify_data import verify


@pytest.fixture
def local_api():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            received.append((self.path, body))
            self.send_response(400 if body['prompt'] == 'fail' else 200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'choices': [{'text': 'नमस्ते hello', 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 10, 'completion_tokens': 3}}).encode())

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{server.server_port}/v1', received
    server.shutdown()
    server.server_close()
    thread.join()


def test_auditkit_raw_greedy_request_and_failed_response(local_api):
    base, received = local_api
    args = SimpleNamespace(model_id='tiny-aya', api_base=base, max_tokens=256, timeout=5)
    row = {'id': 'dev-1', 'text': 'hello', 'lang': 'hinglish'}
    record = one_prompt(row, 'exact-rendered-prompt', args, {'run_fingerprint': 'x', 'template_sha': 't'})
    assert record['response'] == 'नमस्ते hello'
    assert record['answer_script'] == 'devanagari+latin'
    assert record['auditkit_fingerprint']
    endpoint, body = received[0]
    assert endpoint == '/v1/completions'
    assert body['prompt'] == 'exact-rendered-prompt'
    assert body['temperature'] == 0 and body['max_tokens'] == 256 and body['seed'] == 13
    assert body['cache_prompt'] is False
    one_prompt(row, 'exact-rendered-prompt', args, {'run_fingerprint': 'x'}, input_ids=[1, 2, 3])
    assert received[-1][1]['prompt'] == [1, 2, 3]
    with pytest.raises(RuntimeError, match='Generation failed'):
        one_prompt(row, 'fail', args, {'run_fingerprint': 'x'})


def test_resume_preserves_complete_rows_and_recovers_torn_tail(tmp_path):
    path = tmp_path / 'cell.jsonl'
    good = {'prompt_id': 'a', 'run_fingerprint': 'same', 'response': 'hi'}
    path.write_bytes(json.dumps(good).encode() + b'\n{"prompt_id":')
    assert resume_rows(path, 'same') == {'a': good}
    assert path.read_bytes().endswith(b'\n')
    assert path.with_suffix('.partial').read_bytes() == b'{"prompt_id":'
    with pytest.raises(ValueError, match='Resume refused'):
        resume_rows(path, 'changed')
    path.write_text((json.dumps(good) + '\n') * 2)
    with pytest.raises(ValueError, match='Duplicate response'):
        resume_rows(path, 'same')


def test_split_filter_and_duplicate_prompt_rejection(tmp_path):
    path = tmp_path / 'prompts.jsonl'
    rows = [{'id': 'a', 'split': 'dev', 'set': 'benign_general'},
            {'id': 'b', 'split': 'test', 'set': 'harmful_general'},
            {'id': 'c', 'split': 'dev', 'set': 'medmcqa'}]
    path.write_text('\n'.join(map(json.dumps, rows)))
    assert load_rows([path], 'dev') == rows[:1]
    with pytest.raises(ValueError, match='Duplicate prompt'):
        load_rows([path, path], 'dev')


def test_data_package_actual_counts_and_line_endings():
    report = verify(ROOT / 'Datasets/final')
    assert report['evaluation_verified']
    assert report['test_counts'] == {'harmful_general': 550, 'harmful_clinical': 306,
        'benign_general': 450, 'benign_clinical': 306, 'medmcqa': 500}
    assert len(report['errors']) == len(report['crlf_only_mismatches']) == 8


def test_no_preamble_template_preserves_turn_tokens():
    from transformers.utils.chat_template_utils import _compile_jinja_template
    template = _compile_jinja_template((ROOT / 'templates/none.jinja').read_text())
    rendered = template.render(bos_token='<BOS_TOKEN>', messages=[{'role': 'user', 'content': 'hello'}],
                               add_generation_prompt=True)
    assert rendered == ('<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>hello<|END_OF_TURN_TOKEN|>'
                        '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|><|START_RESPONSE|>')
    assert 'preamble' not in rendered
    with pytest.raises(Exception, match='no-preamble experiment'):
        template.render(bos_token='<BOS_TOKEN>', messages=[{'role': 'system', 'content': 'x'}],
                        add_generation_prompt=True)


def test_generation_eos_normalization_preserves_tensor_bytes(tmp_path):
    import numpy as np
    from gguf import GGUFReader, GGUFWriter
    from prepare_models import align_generation_eos
    path = tmp_path / 'model.gguf'
    weights = np.arange(32, dtype=np.float16).reshape(4, 8)
    writer = GGUFWriter(str(path), 'cohere2')
    writer.add_uint32('tokenizer.ggml.eos_token_id', 6)
    writer.add_tensor('token_embd.weight', weights)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    original_size = path.stat().st_size
    assert align_generation_eos(path, 3) == 6
    reader = GGUFReader(str(path))
    field = reader.get_field('tokenizer.ggml.eos_token_id')
    assert int(field.parts[field.data[0]][0]) == 3
    assert reader.tensors[0].data.tobytes() == weights.tobytes()
    assert path.stat().st_size == original_size
