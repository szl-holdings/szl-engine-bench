"""Offline type/completion validation; transport is always an in-memory double."""
import json
import itertools
import unittest
from email.message import Message
from unittest.mock import patch
from szl_engine_bench import engine_bench as engine
DONE = b'data: [DONE]\n\n'

def event(value):
    return ('data: ' + json.dumps(value) + '\n\n').encode()

def text(value):
    return event({'choices': [{'text': value}]})

class Response:

    def __init__(self, lines):
        self.lines = lines
        self.headers = Message()
        self.headers['Content-Type'] = 'text/event-stream'

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.lines)

class StreamProtocolTests(unittest.TestCase):

    def measure(self, lines, sampling=None):
        with patch.object(engine.urllib.request, 'urlopen', return_value=Response(lines)), patch.object(engine.urllib.request, 'build_opener') as builder, patch.object(engine.time, 'perf_counter', side_effect=itertools.count()):
            builder.return_value.open.return_value = Response(lines)
            return engine.measure_once('http://synthetic.invalid', 'fixture', 'fixture', 8, sampling=sampling)

    def test_valid_two_texts(self):
        result = self.measure([text('a'), text('b'), DONE])
        self.assertEqual(result['state'], 'MEASURED')
        self.assertEqual(result['chunks'], 2)

    def test_bound_request_uses_same_protocol_validation(self):
        sampling = {'temperature': 0, 'top_p': 1, 'seed': 7}
        self.assertEqual(self.measure([text('a'), text('b'), DONE], sampling)['chunks'], 2)
        with self.assertRaisesRegex(ValueError, 'text must be a string'):
            self.measure([text('a'), text(17), DONE], sampling)

    def test_metadata_empty_text_not_counted(self):
        result = self.measure([event({'choices': [{}]}), text(''), text('a'), event({'choices': [], 'usage': {'completion_tokens': 2}}), text('b'), DONE])
        self.assertEqual(result['chunks'], 2)
        self.assertEqual(result['completion_tokens'], 2)

    def test_no_text_never_measured(self):
        self.assertEqual(self.measure([event({'choices': []}), DONE])['state'], 'INVALID')

    def test_truncated_eof(self):
        with self.assertRaisesRegex(ValueError, 'before the \\[DONE\\]'):
            self.measure([text('a'), text('b')])

    def test_invalid_json(self):
        with self.assertRaisesRegex(ValueError, 'invalid JSON'):
            self.measure([text('a'), b'data: {bad}\n', DONE])

    def test_provider_error(self):
        with self.assertRaisesRegex(ValueError, 'provider error'):
            self.measure([text('a'), text('b'), event({'error': {'message': 'private'}}), DONE])

def install_case(name, payload, error):

    def case(self):
        with self.assertRaisesRegex(ValueError, error):
            self.measure([text('a'), event(payload), text('b'), DONE])
    setattr(StreamProtocolTests, 'test_' + name, case)
for name, value in [('number', 17), ('bool', True), ('list', ['bad']), ('dict', {'bad': 1}), ('null', None), ('false', False), ('zero', 0), ('empty_list', [])]:
    install_case('text_' + name, {'choices': [{'text': value}]}, 'text must be a string')
for name, value in [('null', None), ('dict', {}), ('bool', False), ('number', 17), ('string', 'bad')]:
    install_case('choices_' + name, {'choices': value}, 'choices must be an array')
for name, value in [('null', None), ('list', []), ('bool', False), ('number', 17), ('string', 'bad')]:
    install_case('choice_' + name, {'choices': [value]}, 'choice must be an object')
install_case('second_choice_invalid', {'choices': [{'text': 'valid'}, {'text': 17}]}, 'text must be a string')
if __name__ == '__main__':
    unittest.main()
