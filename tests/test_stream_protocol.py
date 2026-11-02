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

    def test_duplicate_json_fields_are_rejected_in_both_transport_paths(self):
        payloads = (
            b'data: {"choices":[{"text":"shadow"}],"choices":[{"text":"b"}]}\n\n',
            b'data: {"choices":[{"text":"shadow","text":"b"}]}\n\n',
        )
        for sampling in (None, {"temperature": 0, "top_p": 1, "seed": 7}):
            for payload in payloads:
                with self.subTest(sampling=sampling, payload=payload):
                    with self.assertRaisesRegex(ValueError, 'invalid JSON'):
                        self.measure([text('a'), payload, DONE], sampling)

    def test_nonfinite_json_numbers_are_rejected_in_both_transport_paths(self):
        for constant in ("NaN", "Infinity", "-Infinity", "1e999"):
            payload = (
                f'data: {{"choices":[{{"text":"b"}}],"ignored":{constant}}}\n\n'
            ).encode()
            for sampling in (None, {"temperature": 0, "top_p": 1, "seed": 7}):
                with self.subTest(constant=constant, sampling=sampling):
                    with self.assertRaisesRegex(ValueError, 'invalid JSON'):
                        self.measure([text('a'), payload, DONE], sampling)

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
class UsageShapeTests(unittest.TestCase):
    measure = StreamProtocolTests.measure

    def test_non_object_usage_is_rejected_in_both_transport_paths(self):
        for value in (False, True, 0, 17, [], [1], "bad"):
            for sampling in (None, {"temperature": 0, "top_p": 1, "seed": 7}):
                for before_text in (False, True):
                    with self.subTest(value=value, sampling=sampling, before_text=before_text):
                        usage = event({"choices": [], "usage": value})
                        chunks = [usage, text("a"), text("b")] if before_text else [text("a"), text("b"), usage]
                        with self.assertRaisesRegex(ValueError, "usage must be an object"):
                            self.measure(chunks + [DONE], sampling)

    def test_optional_and_valid_usage_preserve_unknown_or_reported_counts(self):
        for payload, expected in (({}, None), ({"usage": None}, None), ({"usage": {}}, None),
                                  ({"usage": {"completion_tokens": None}}, None),
                                  ({"usage": {"completion_tokens": 0}}, 0),
                                  ({"usage": {"completion_tokens": 2}}, 2)):
            with self.subTest(payload=payload):
                result = self.measure([text("a"), event({"choices": [], **payload}), text("b"), DONE])
                self.assertEqual(result["state"], "MEASURED")
                self.assertEqual(result["chunks"], 2)
                self.assertEqual(result["completion_tokens"], expected)

if __name__ == '__main__':
    unittest.main()
