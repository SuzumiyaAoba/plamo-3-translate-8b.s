"""Range checks must never read a full multi-gigabyte response by accident."""
import io
import json
import struct
import unittest
from unittest.mock import patch
from tools.model_catalog import profile
from tools.safetensors_headers import fetch_header


class Response(io.BytesIO):
    def __init__(self, data, end, status=206):
        super().__init__(data)
        self.status = status
        self.headers = {'Content-Range': f'bytes 0-{end}/{profile("0.8b").manifest["files"]["model.safetensors"]["size"]}'}
        self.read_called = False

    def read(self, *args):
        self.read_called = True
        return super().read(*args)


class HeaderTests(unittest.TestCase):
    def test_valid_bounded_ranges(self):
        header = json.dumps({'x': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
        data = struct.pack('<Q', len(header)) + header
        replies = [Response(data[:8], 7), Response(data, len(data) - 1)]
        with patch('tools.safetensors_headers.urllib.request.urlopen', side_effect=replies) as fetch:
            self.assertEqual(fetch_header(profile('0.8b'), 'model.safetensors'), data)
        self.assertEqual(fetch.call_args_list[0].args[0].get_header('Range'), 'bytes=0-7')

    def test_ignored_range_is_rejected_without_reading_payload(self):
        response = Response(b'large payload', 7, status=200)
        with patch('tools.safetensors_headers.urllib.request.urlopen', return_value=response):
            with self.assertRaisesRegex(ValueError, 'range'):
                fetch_header(profile('0.8b'), 'model.safetensors')
        self.assertFalse(response.read_called)

    def test_excessive_header_does_not_trigger_second_request(self):
        with patch('tools.safetensors_headers.urllib.request.urlopen', return_value=Response(struct.pack('<Q', 2**40), 7)) as fetch:
            with self.assertRaisesRegex(ValueError, 'length'):
                fetch_header(profile('0.8b'), 'model.safetensors')
            fetch.assert_called_once()

    def test_truncated_header_is_rejected(self):
        with patch('tools.safetensors_headers.urllib.request.urlopen', return_value=Response(b'123', 7)):
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                fetch_header(profile('0.8b'), 'model.safetensors')


if __name__ == '__main__':
    unittest.main()
