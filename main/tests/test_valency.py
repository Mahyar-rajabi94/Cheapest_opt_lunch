"""Offline literature tests: real MCP framing, fake papers, no objective calls."""
import asyncio
from contextlib import asynccontextmanager
import json
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.blind_runner import valency

try:
    from mcp import Client, StdioServerParameters
    from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool
    SDK = True
except ImportError:
    SDK = False


def definitions():
    return [Tool(name='semantic_search_papers', description='Find papers by meaning',
                 input_schema={'type': 'object', 'properties': {'query': {'type': 'string'}},
                               'required': ['query'], 'additionalProperties': False}),
            Tool(name='get_paper_by_id', input_schema={'type': 'object',
                 'properties': {'paper_id': {'type': 'string'}}, 'required': ['paper_id']}),
            Tool(name='list_sources', input_schema={'type': 'object'})]


def evidence(text='An abstract about optimizer adaptation.'):
    return CallToolResult(content=[TextContent(type='text', text=text)],
                          structured_content={'papers': [{'id': 'fixture:001',
                              'title': 'Test optimization paper', 'abstract': text}]})


def unpack(result):
    return json.loads(result.content[0].text)


@unittest.skipUnless(SDK, 'Install main/requirements-literature.txt for literature tests')
class GatewayTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.config = {'max_calls': 2, 'timeout_seconds': 1}
        self.fetch = AsyncMock(return_value=evidence())
        self.schemas = {t.name: t.input_schema for t in definitions()}
        self.gateway = valency.Gateway(self.config, self.directory, self.schemas, self.fetch)

    def tearDown(self):
        self.temp.cleanup()

    async def test_caches_and_records_exact_evidence(self):
        a = await self.gateway.call('semantic_search_papers', {'query': 'noise'})
        b = await self.gateway.call('semantic_search_papers', {'query': 'noise'})
        self.assertEqual(a, b)
        self.fetch.assert_awaited_once()
        raw = json.loads(unpack(a)['evidence_json'])
        self.assertEqual(raw['structuredContent']['papers'][0]['id'], 'fixture:001')
        events = [json.loads(x) for x in self.gateway.journal.read_text().splitlines()]
        self.assertEqual([e['event'] for e in events], ['request', 'response', 'cache_hit'])
        self.assertEqual(events[0]['arguments'], {'query': 'noise'})
        saved = self.gateway.responses / (unpack(a)['retrieval_id'] + '.json')
        self.assertEqual(events[1]['sha256'], hashlib.sha256(saved.read_bytes()).hexdigest())

    async def test_disallowed_tool_and_invalid_arguments_never_reach_network(self):
        for name, args in [('submit_feedback', {}), ('fetch_url', {'url': 'https://example.org'}),
                           ('semantic_search_papers', {}),
                           ('semantic_search_papers', {'query': 'x', 'url': 'https://example.org'}),
                           ('semantic_search_papers', {'query': 'x' * 16001})]:
            self.assertTrue((await self.gateway.call(name, args)).is_error)
        self.fetch.assert_not_awaited()
        self.assertEqual(self.gateway.used, 0)

    async def test_concurrent_calls_share_limit_and_cache(self):
        results = await asyncio.gather(*[self.gateway.call('semantic_search_papers', {'query': q})
                                        for q in ['a', 'a', 'b', 'c']])
        self.assertEqual(self.fetch.await_count, 2)
        self.assertTrue(results[-1].is_error)
        self.assertEqual(self.gateway.used, 2)

    async def test_limit_and_cache_survive_restart(self):
        for q in ['a', 'b']:
            await self.gateway.call('semantic_search_papers', {'query': q})
        restarted = valency.Gateway(self.config, self.directory, self.schemas, self.fetch)
        self.assertFalse((await restarted.call('semantic_search_papers', {'query': 'a'})).is_error)
        self.assertTrue((await restarted.call('semantic_search_papers', {'query': 'c'})).is_error)
        self.assertEqual(self.fetch.await_count, 2)

    async def test_independent_queries_do_not_wait_behind_other_network_calls(self):
        both_started = asyncio.Event()
        count = 0
        async def fetch(*args):
            nonlocal count
            count += 1
            if count == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), .2)
            return evidence()
        self.fetch.side_effect = fetch
        results = await asyncio.gather(*[self.gateway.call('semantic_search_papers', {'query': q})
                                        for q in ['a', 'b']])
        self.assertTrue(all(not result.is_error for result in results))

    async def test_cancelled_caller_does_not_cancel_shared_retrieval(self):
        entered, finish = asyncio.Event(), asyncio.Event()
        async def fetch(*args):
            entered.set()
            await finish.wait()
            return evidence()
        self.fetch.side_effect = fetch
        a = asyncio.create_task(self.gateway.call('semantic_search_papers', {'query': 'a'}))
        await entered.wait()
        a.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await a
        b = asyncio.create_task(self.gateway.call('semantic_search_papers', {'query': 'a'}))
        finish.set()
        self.assertFalse((await b).is_error)
        self.fetch.assert_awaited_once()

    async def test_timeout_returns_control_and_charges_attempt(self):
        async def slow(*args):
            await asyncio.sleep(1)
        self.gateway.config['timeout_seconds'] = .01
        self.fetch.side_effect = slow
        result = await self.gateway.call('semantic_search_papers', {'query': 'a'})
        self.assertTrue(result.is_error)
        self.assertEqual(self.gateway.used, 1)
        self.assertIn('continue optimization', result.content[0].text)
        self.fetch.side_effect = None
        self.assertFalse((await self.gateway.call('semantic_search_papers', {'query': 'a'})).is_error)

    async def test_remote_errors_do_not_leak_transport_secrets(self):
        self.fetch.side_effect = RuntimeError('Authorization: Bearer SECRET')
        result = await self.gateway.call('semantic_search_papers', {'query': 'a'})
        self.assertTrue(result.is_error)
        self.assertNotIn('SECRET', result.model_dump_json())
        self.assertNotIn('SECRET', self.gateway.journal.read_text())
        self.fetch.side_effect = None
        self.fetch.return_value = CallToolResult(is_error=True,
            content=[TextContent(type='text', text='remote failure with SECRET')])
        self.assertTrue((await self.gateway.call('semantic_search_papers', {'query': 'b'})).is_error)
        self.assertNotIn('SECRET', self.gateway.journal.read_text())

    async def test_pages_preserve_complete_result_without_extra_requests(self):
        self.fetch.return_value = evidence('a' * (valency.PAGE_CHARS * 2))
        first = unpack(await self.gateway.call('semantic_search_papers', {'query': 'large'}))
        pieces, page = [], first
        while True:
            pieces.append(page['evidence_json'])
            if page['next_offset'] is None:
                break
            page = unpack(await self.gateway.call(valency.PAGE_TOOL, {
                'retrieval_id': first['retrieval_id'], 'offset': page['next_offset']}))
        self.assertEqual(json.loads(''.join(pieces)),
                         self.fetch.return_value.model_dump(by_alias=True, exclude_none=True))
        self.fetch.assert_awaited_once()
        for key, offset in [('../../config', 0), (first['retrieval_id'], -1),
                            (first['retrieval_id'], True), (first['retrieval_id'], 99999999)]:
            self.assertTrue((await self.gateway.call(valency.PAGE_TOOL,
                {'retrieval_id': key, 'offset': offset})).is_error)

    async def test_real_mcp_client_sees_only_selected_tools(self):
        raw = [t.model_dump(by_alias=True, exclude_none=True) for t in definitions()]
        server = valency.make_server(self.gateway, raw)
        async with Client(server, mode='legacy') as client:
            names = {t.name for t in (await client.list_tools()).tools}
            self.assertEqual(names, set(self.schemas) | {valency.PAGE_TOOL})
            result = await client.call_tool('semantic_search_papers', {'query': 'noise'})
            self.assertIn('fixture:001', unpack(result)['evidence_json'])
            self.assertTrue((await client.call_tool('submit_feedback', {})).is_error)

    async def test_real_stdio_transport(self):
        params = StdioServerParameters(command=sys.executable,
            args=[str(Path(__file__).resolve()), '--fixture-server', str(self.directory)])
        async with Client(params, mode='legacy') as client:
            result = await client.call_tool('get_paper_by_id', {'paper_id': 'fixture:001'})
            self.assertIn('fixture:001', unpack(result)['evidence_json'])
            self.assertEqual(len((await client.list_tools()).tools), 4)


@unittest.skipUnless(SDK, 'Install main/requirements-literature.txt for literature tests')
class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_http_redirect_is_not_followed(self):
        import httpx2
        real_client = httpx2.AsyncClient
        destinations = []
        def redirect(request):
            destinations.append(str(request.url))
            return httpx2.Response(307, headers={'Location': 'https://example.org/forbidden'})
        def http(**kwargs):
            return real_client(**kwargs, transport=httpx2.MockTransport(redirect))
        with patch.dict(os.environ, {valency.TOKEN_ENV: 'test-token'}), patch.object(httpx2, 'AsyncClient', http):
            with self.assertRaises(Exception):
                async with asyncio.timeout(1):
                    async with valency.connect({'timeout_seconds': .1}):
                        self.fail('Redirected connection should not initialize')
        self.assertEqual(destinations, [valency.URL])

    async def test_discovery_paginates_filters_and_snapshots_real_schemas(self):
        pages = [ListToolsResult(tools=[*definitions()[:1], Tool(name='submit_feedback',
                    input_schema={'type': 'object'})], next_cursor='page2'),
                 ListToolsResult(tools=definitions()[1:])]
        client = type('FakeClient', (), {'list_tools': AsyncMock(side_effect=pages),
                     'call_tool': AsyncMock(return_value=evidence()), 'protocol_version': '2025-11-25'})()
        @asynccontextmanager
        async def connector(config):
            yield client
        with tempfile.TemporaryDirectory() as tmp:
            result = await valency.discover({'timeout_seconds': 1}, Path(tmp), connector)
            self.assertEqual({t['name'] for t in result['tools']}, {t.name for t in definitions()})
            client.list_tools.assert_any_await(cursor='page2')
            client.call_tool.assert_awaited_once_with('list_sources', {})
            self.assertTrue((Path(tmp) / 'connection_check.json').exists())

    async def test_http_handshake_uses_fixed_destination_and_bearer(self):
        import httpx2
        real_client = httpx2.AsyncClient
        requests = []
        def respond(request):
            requests.append(request)
            self.assertEqual(str(request.url), valency.URL)
            self.assertEqual(request.headers['Authorization'], 'Bearer test-token')
            if request.method == 'DELETE':
                return httpx2.Response(200)
            if request.method == 'GET':
                return httpx2.Response(405)
            data = json.loads(request.content)
            if data['method'] == 'notifications/initialized':
                return httpx2.Response(202)
            if data['method'] == 'initialize':
                result = {'protocolVersion': '2025-11-25', 'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'fixture', 'version': '1'}}
            elif data['method'] == 'tools/list':
                result = {'tools': [t.model_dump(by_alias=True, exclude_none=True) for t in definitions()]}
            else:
                result = evidence().model_dump(by_alias=True, exclude_none=True)
            return httpx2.Response(200, json={'jsonrpc': '2.0', 'id': data['id'], 'result': result})
        def http(**kwargs):
            self.assertFalse(kwargs['follow_redirects'])
            self.assertFalse(kwargs['trust_env'])
            return real_client(**kwargs, transport=httpx2.MockTransport(respond))
        with patch.dict(os.environ, {valency.TOKEN_ENV: 'test-token'}), patch.object(httpx2, 'AsyncClient', http):
            async with valency.connect({'timeout_seconds': 1}) as client:
                self.assertFalse((await client.call_tool('list_sources', {})).is_error)
        self.assertGreaterEqual(len(requests), 3)
        self.assertIn('mcp-protocol-version', requests[-1].headers)


def fixture_server(directory):
    from mcp.server.stdio import stdio_server
    async def run():
        async def fetch(name, arguments):
            return evidence()
        tools = definitions()
        gateway = valency.Gateway({'max_calls': 2, 'timeout_seconds': 1}, directory,
                                  {t.name: t.input_schema for t in tools}, fetch)
        server = valency.make_server(gateway, [t.model_dump(by_alias=True) for t in tools])
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    asyncio.run(run())


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--fixture-server':
        fixture_server(Path(sys.argv[2]))
    else:
        unittest.main()
