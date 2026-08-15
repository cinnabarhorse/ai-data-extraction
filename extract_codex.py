#!/usr/bin/env python3
"""Extract Codex conversations and tool activity from local rollout files."""

import argparse
import json
import os
import platform
from datetime import datetime
from pathlib import Path


def find_codex_installations():
    """Find Codex data directories without reading configuration or auth files."""
    system = platform.system()
    home = Path.home()
    patterns = ('codex', 'codex-local', '.codex', '.codex-local')

    if system == 'Darwin':
        bases = (home / 'Library/Application Support', home / '.config', home)
    elif system == 'Linux':
        bases = (home / '.config', home / '.local/share', home)
    elif system == 'Windows':
        bases = (
            Path(os.environ.get('APPDATA', str(home / 'AppData/Roaming'))),
            Path(os.environ.get('LOCALAPPDATA', str(home / 'AppData/Local'))),
            home,
        )
    else:
        bases = (home / '.config', home)

    return sorted({base / pattern
                   for base in bases if base.exists()
                   for pattern in patterns if (base / pattern).exists()})


def find_all_codex_sessions(installation):
    """Find rollout files in both date- and project-organized stores."""
    files = set()
    sessions_dir = installation / 'sessions'
    projects_dir = installation / 'projects'
    if sessions_dir.exists():
        files.update(sessions_dir.rglob('rollout-*.jsonl'))
    if projects_dir.exists():
        files.update(projects_dir.rglob('*.jsonl'))
    return sorted(files)


def is_heartbeat_message(text):
    """Return whether a user message is an automation heartbeat envelope."""
    return isinstance(text, str) and text.lstrip().lower().startswith('<heartbeat>')


def _message_text(content):
    """Flatten modern Responses-style text blocks without copying image data."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ''

    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get('text')
            if isinstance(text, str):
                parts.append(text)
    return '\n'.join(parts).strip()


def _json_value(value):
    """Decode JSON arguments when possible while preserving plain text."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def _source_kind(source):
    if isinstance(source, dict) and 'subagent' in source:
        return 'subagent'
    if isinstance(source, str):
        return source
    return None


def _modern_tool_event(payload, timestamp, turn_index, pending_tools):
    """Convert a current response_item tool payload to the legacy flat schema."""
    payload_type = payload.get('type')
    call_id = payload.get('call_id')

    if payload_type in ('custom_tool_call', 'function_call'):
        tool = payload.get('name')
        if call_id:
            pending_tools[call_id] = tool
        raw_input = payload.get('input') if payload_type == 'custom_tool_call' \
            else payload.get('arguments')
        return {
            'type': 'tool_use',
            'tool': tool,
            'input': _json_value(raw_input),
            'call_id': call_id,
            'provider_type': payload_type,
            'turn_index': turn_index,
            'timestamp': timestamp,
        }

    if payload_type in ('custom_tool_call_output', 'function_call_output'):
        return {
            'type': 'tool_result',
            'tool': pending_tools.get(call_id),
            'output': _json_value(payload.get('output')),
            'call_id': call_id,
            'provider_type': payload_type,
            'turn_index': turn_index,
            'timestamp': timestamp,
        }

    if payload_type == 'web_search_call':
        return {
            'type': 'tool_use',
            'tool': 'web_search',
            'input': payload.get('action'),
            'call_id': payload.get('id'),
            'provider_type': payload_type,
            'turn_index': turn_index,
            'timestamp': timestamp,
        }

    if payload_type == 'tool_search_call':
        if call_id:
            pending_tools[call_id] = 'tool_search'
        return {
            'type': 'tool_use',
            'tool': 'tool_search',
            'input': _json_value(payload.get('arguments')),
            'call_id': call_id,
            'provider_type': payload_type,
            'turn_index': turn_index,
            'timestamp': timestamp,
        }

    if payload_type == 'tool_search_output':
        return {
            'type': 'tool_result',
            'tool': pending_tools.get(call_id, 'tool_search'),
            'output': payload.get('tools'),
            'call_id': call_id,
            'provider_type': payload_type,
            'turn_index': turn_index,
            'timestamp': timestamp,
        }

    return None


def extract_codex_session(session_file, exclude_heartbeats=False,
                          include_tools=True):
    """Extract one rollout, supporting both legacy and current Codex schemas."""
    messages = []
    tool_results = []
    session_meta = None
    pending_tools = {}
    pending_user = None
    awaiting_user = False
    turn_index = -1
    skip_turn = False
    heartbeat_turns = 0

    def add_user(text, timestamp, context=None):
        nonlocal turn_index, skip_turn, heartbeat_turns
        turn_index += 1
        skip_turn = exclude_heartbeats and is_heartbeat_message(text)
        if skip_turn:
            heartbeat_turns += 1
            return
        message = {
            'role': 'user',
            'content': text.strip(),
            'turn_index': turn_index,
            'timestamp': timestamp,
        }
        if context is not None:
            message['context'] = context
        messages.append(message)

    def flush_pending_user():
        nonlocal pending_user, awaiting_user
        if pending_user:
            add_user(pending_user['content'], pending_user['timestamp'])
        pending_user = None
        awaiting_user = False

    with open(session_file, 'r', encoding='utf-8') as handle:
        for line in handle:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            event_type = obj.get('type')
            payload = obj.get('payload', {})
            timestamp = obj.get('timestamp')

            if event_type == 'session_meta':
                # Later metadata records describe resumes/forks and must not replace
                # the unique rollout identity from the first record.
                if session_meta is None:
                    session_meta = payload
                continue

            if event_type == 'turn_context':
                flush_pending_user()
                awaiting_user = True
                continue

            if event_type == 'event_msg':
                payload_type = payload.get('type')
                if payload_type == 'user_message':
                    text = (payload.get('message') or '').strip()
                    if text:
                        pending_user = None
                        awaiting_user = False
                        add_user(text, timestamp, payload.get('context'))
                    continue

                flush_pending_user()
                if payload_type == 'agent_message':
                    text = (payload.get('message') or '').strip()
                    if text and not skip_turn:
                        message = {
                            'role': 'assistant',
                            'content': text,
                            'turn_index': turn_index,
                            'timestamp': timestamp,
                        }
                        if payload.get('model'):
                            message['model'] = payload['model']
                        if payload.get('phase'):
                            message['phase'] = payload['phase']
                        messages.append(message)
                elif include_tools and not skip_turn and payload_type in (
                        'tool_use', 'tool_result', 'diff'):
                    if payload_type == 'tool_use':
                        tool_results.append({
                            'type': 'tool_use',
                            'tool': payload.get('tool'),
                            'input': payload.get('input'),
                            'turn_index': turn_index,
                            'timestamp': timestamp,
                        })
                    elif payload_type == 'tool_result':
                        tool_results.append({
                            'type': 'tool_result',
                            'tool': payload.get('tool'),
                            'output': payload.get('output'),
                            'turn_index': turn_index,
                            'timestamp': timestamp,
                        })
                    else:
                        tool_results.append({
                            'type': 'diff',
                            'file': payload.get('file'),
                            'diff': payload.get('diff'),
                            'turn_index': turn_index,
                            'timestamp': timestamp,
                        })
                continue

            if event_type != 'response_item':
                continue

            payload_type = payload.get('type')
            if payload_type == 'message' and payload.get('role') == 'user':
                text = _message_text(payload.get('content'))
                if awaiting_user and text:
                    pending_user = {'content': text, 'timestamp': timestamp}
                continue

            flush_pending_user()
            if payload_type == 'message' and payload.get('role') == 'assistant':
                text = _message_text(payload.get('content'))
                if not text or skip_turn:
                    continue
                phase = payload.get('phase')
                match = next((message for message in reversed(messages)
                              if message.get('turn_index') == turn_index
                              and message['role'] == 'assistant'
                              and message['content'] == text), None)
                if match:
                    if phase:
                        match['phase'] = phase
                else:
                    message = {
                        'role': 'assistant',
                        'content': text,
                        'turn_index': turn_index,
                        'timestamp': timestamp,
                    }
                    if phase:
                        message['phase'] = phase
                    messages.append(message)
                continue

            if include_tools and not skip_turn:
                tool_event = _modern_tool_event(
                    payload, timestamp, turn_index, pending_tools)
                if tool_event:
                    tool_results.append(tool_event)

    flush_pending_user()
    if not messages and not tool_results:
        return None

    session_meta = session_meta or {}
    conversation = {
        'messages': messages,
        'session_id': session_meta.get('id'),
        'root_session_id': session_meta.get('session_id') or session_meta.get('id'),
        'cwd': session_meta.get('cwd'),
        'source': 'codex',
        'source_kind': _source_kind(session_meta.get('source')),
        'originator': session_meta.get('originator'),
        'session_file': str(session_file),
        'timestamp': session_meta.get('timestamp'),
    }
    for key in ('parent_thread_id', 'forked_from_id', 'agent_nickname'):
        if session_meta.get(key):
            conversation[key] = session_meta[key]
    if tool_results:
        conversation['tool_results'] = tool_results
    if heartbeat_turns:
        conversation['heartbeat_turns_excluded'] = heartbeat_turns
    return conversation


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='extracted_data',
                        help='Directory for the timestamped JSONL export')
    parser.add_argument('--exclude-heartbeats', action='store_true',
                        help='Remove heartbeat prompts and their complete turns')
    parser.add_argument('--messages-only', action='store_true',
                        help='Skip tool calls/outputs for a compact chat export')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most this many rollout files (0 = all)')
    return parser.parse_args()


def main():
    args = _arguments()
    print('=' * 80)
    print('CODEX COMPLETE DATA EXTRACTION')
    print('=' * 80)
    print()

    print('🔍 Searching for Codex installations...')
    installations = find_codex_installations()
    if not installations:
        print('❌ No Codex installations found!')
        return

    print(f'✅ Found {len(installations)} installation(s):')
    for installation in installations:
        print(f'   - {installation}')
    print()

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    if os.name != 'nt':
        os.chmod(str(output_dir), 0o700)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = output_dir / f'codex_conversations_{timestamp}.jsonl'

    total = complete = total_messages = with_tools = tool_events = 0
    heartbeat_turns = processed = 0
    installation_stats = {}

    with open(output_file, 'w', encoding='utf-8') as output:
        for installation in installations:
            session_files = find_all_codex_sessions(installation)
            print(f'📂 Processing: {installation}')
            print(f'   Found {len(session_files)} session files')
            count = 0
            for session_file in session_files:
                if args.limit and processed >= args.limit:
                    break
                processed += 1
                try:
                    conversation = extract_codex_session(
                        session_file,
                        exclude_heartbeats=args.exclude_heartbeats,
                        include_tools=not args.messages_only,
                    )
                except OSError as error:
                    print(f'   ⚠️  Skipped {session_file}: {error}')
                    continue
                if not conversation:
                    continue
                conversation['installation'] = str(installation)
                output.write(json.dumps(conversation, ensure_ascii=False) + '\n')
                count += 1
                total += 1
                message_count = len(conversation['messages'])
                total_messages += message_count
                complete += any(message['role'] == 'assistant'
                                for message in conversation['messages'])
                events = conversation.get('tool_results', [])
                with_tools += bool(events)
                tool_events += len(events)
                heartbeat_turns += conversation.get('heartbeat_turns_excluded', 0)
            installation_stats[str(installation)] = count
            print(f'   ✅ {count} conversations')
            if args.limit and processed >= args.limit:
                break

    if os.name != 'nt':
        os.chmod(str(output_file), 0o600)

    print()
    print('=' * 80)
    print('EXTRACTION COMPLETE')
    print('=' * 80)
    print(f'Total conversations: {total:,}')
    print(f'Complete conversations: {complete:,}')
    print(f'Total messages: {total_messages:,}')
    print(f'With tool use/diffs: {with_tools:,}')
    print(f'Tool call/result events: {tool_events:,}')
    if args.exclude_heartbeats:
        print(f'Heartbeat turns excluded: {heartbeat_turns:,}')
    print()
    print('Breakdown by installation:')
    for installation, count in sorted(
            installation_stats.items(), key=lambda item: -item[1]):
        print(f'  {Path(installation).name:20} {count:5,} conversations')
    print()
    size = output_file.stat().st_size / 1024 / 1024
    print(f'✅ Saved to: {output_file}')
    print(f'   Size: {size:.2f} MB')
    print('   Format: JSONL (one conversation per line)')


if __name__ == '__main__':
    main()
