import json
import tempfile
import unittest
from pathlib import Path

from extract_codex import extract_codex_session, is_heartbeat_message


def record(kind, payload, timestamp='2026-08-15T00:00:00Z'):
    return {'timestamp': timestamp, 'type': kind, 'payload': payload}


class ExtractCodexTest(unittest.TestCase):
    def write_rollout(self, records):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / 'rollout-test.jsonl'
        path.write_text(''.join(json.dumps(item) + '\n' for item in records),
                        encoding='utf-8')
        self.addCleanup(directory.cleanup)
        return path

    def test_modern_tools_lineage_phases_and_heartbeat_filter(self):
        heartbeat = '<heartbeat><automation_id>noise</automation_id></heartbeat>'
        records = [
            record('session_meta', {
                'id': 'rollout-unique',
                'session_id': 'root-thread',
                'cwd': '/tmp/project',
                'timestamp': '2026-08-15T00:00:00Z',
                'originator': 'Codex Desktop',
                'source': {'subagent': {'thread_spawn': {}}},
                'parent_thread_id': 'parent-thread',
            }),
            # Bootstrap context is not a real turn and must not be exported.
            record('response_item', {
                'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': 'bootstrap'}],
            }),
            record('turn_context', {}),
            record('response_item', {
                'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': 'Fix it'}],
            }),
            record('event_msg', {'type': 'user_message', 'message': 'Fix it'}),
            record('response_item', {
                'type': 'custom_tool_call', 'name': 'exec_command',
                'input': '{"cmd":"pytest"}', 'call_id': 'custom-1',
            }),
            record('response_item', {
                'type': 'custom_tool_call_output',
                'output': '{"exit_code":0}', 'call_id': 'custom-1',
            }),
            record('response_item', {
                'type': 'reasoning', 'summary': 'private reasoning'
            }),
            record('event_msg', {
                'type': 'agent_message', 'message': 'Running verification.'
            }),
            record('response_item', {
                'type': 'message', 'role': 'assistant', 'phase': 'commentary',
                'content': [{'type': 'output_text',
                             'text': 'Running verification.'}],
            }),
            record('response_item', {
                'type': 'function_call', 'name': 'apply_patch',
                'arguments': '{"patch":"safe"}', 'call_id': 'function-1',
            }),
            record('response_item', {
                'type': 'function_call_output', 'output': 'ok',
                'call_id': 'function-1',
            }),
            record('event_msg', {
                'type': 'agent_message', 'message': 'Done and tested.'
            }),
            record('response_item', {
                'type': 'message', 'role': 'assistant', 'phase': 'final_answer',
                'content': [{'type': 'output_text', 'text': 'Done and tested.'}],
            }),
            # A later resume record used to overwrite the unique rollout ID.
            record('session_meta', {'id': 'shared-parent-id'}),
            record('turn_context', {}),
            record('response_item', {
                'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': heartbeat}],
            }),
            record('event_msg', {'type': 'user_message', 'message': heartbeat}),
            record('response_item', {
                'type': 'function_call', 'name': 'noisy_tool',
                'arguments': '{}', 'call_id': 'heartbeat-tool',
            }),
            record('event_msg', {
                'type': 'agent_message', 'message': 'Noisy automation result.'
            }),
            record('turn_context', {}),
            record('response_item', {
                'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': 'One more thing'}],
            }),
            record('event_msg', {
                'type': 'user_message', 'message': 'One more thing'
            }),
            record('event_msg', {
                'type': 'agent_message', 'message': 'Next result.'
            }),
        ]
        conversation = extract_codex_session(
            self.write_rollout(records), exclude_heartbeats=True)

        self.assertEqual(conversation['session_id'], 'rollout-unique')
        self.assertEqual(conversation['root_session_id'], 'root-thread')
        self.assertEqual(conversation['source_kind'], 'subagent')
        self.assertEqual(conversation['heartbeat_turns_excluded'], 1)
        self.assertEqual(
            [message['content'] for message in conversation['messages']],
            ['Fix it', 'Running verification.', 'Done and tested.',
             'One more thing', 'Next result.'],
        )
        self.assertEqual(conversation['messages'][1]['phase'], 'commentary')
        self.assertEqual(conversation['messages'][2]['phase'], 'final_answer')
        self.assertEqual([event['tool'] for event in conversation['tool_results']],
                         ['exec_command', 'exec_command',
                          'apply_patch', 'apply_patch'])
        self.assertEqual(conversation['tool_results'][0]['input'],
                         {'cmd': 'pytest'})
        self.assertNotIn('heartbeat-tool', json.dumps(conversation))
        self.assertNotIn('private reasoning', json.dumps(conversation))

    def test_messages_only_omits_modern_tools(self):
        records = [
            record('session_meta', {'id': 'one'}),
            record('event_msg', {'type': 'user_message', 'message': 'Run it'}),
            record('response_item', {
                'type': 'function_call', 'name': 'shell',
                'arguments': '{}', 'call_id': 'call-1',
            }),
            record('event_msg', {
                'type': 'agent_message', 'message': 'Complete.'
            }),
        ]
        conversation = extract_codex_session(
            self.write_rollout(records), include_tools=False)
        self.assertNotIn('tool_results', conversation)

    def test_legacy_tool_events_remain_supported(self):
        records = [
            record('session_meta', {'id': 'legacy'}),
            record('event_msg', {'type': 'user_message', 'message': 'Test'}),
            record('event_msg', {
                'type': 'tool_use', 'tool': 'shell', 'input': {'cmd': 'true'}
            }),
            record('event_msg', {
                'type': 'tool_result', 'tool': 'shell', 'output': 'ok'
            }),
            record('event_msg', {
                'type': 'agent_message', 'message': 'Passed.'
            }),
        ]
        conversation = extract_codex_session(self.write_rollout(records))
        self.assertEqual(len(conversation['tool_results']), 2)
        self.assertEqual(conversation['tool_results'][0]['turn_index'], 0)

    def test_response_user_is_fallback_when_event_message_is_missing(self):
        records = [
            record('session_meta', {'id': 'fallback'}),
            record('turn_context', {}),
            record('response_item', {
                'type': 'message', 'role': 'user',
                'content': [{'type': 'input_text', 'text': 'Fallback prompt'}],
            }),
            record('response_item', {
                'type': 'message', 'role': 'assistant', 'phase': 'final_answer',
                'content': [{'type': 'output_text', 'text': 'Fallback answer'}],
            }),
        ]
        conversation = extract_codex_session(self.write_rollout(records))
        self.assertEqual([message['role'] for message in conversation['messages']],
                         ['user', 'assistant'])
        self.assertEqual(conversation['messages'][0]['content'], 'Fallback prompt')

    def test_heartbeat_detection_ignores_leading_whitespace_and_case(self):
        self.assertTrue(is_heartbeat_message('  <HEARTBEAT>test'))
        self.assertFalse(is_heartbeat_message('keep the heartbeat healthy'))


if __name__ == '__main__':
    unittest.main()
