const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(resolve(__dirname, '../index.js'), 'utf8');
const hash = text => [...text].reduce((value, char) => (value * 31 + char.charCodeAt(0)) | 0, 0);

// Load the real extension with only its SillyTavern/browser boundaries replaced.
// Its ready callback is skipped because these tests exercise retrieval and injection.
async function loadExtension({ scores = [0.51, 0.95, 0.5, 0.1, 0.75], mode = 'probability',
    threshold = 0.5, ignoreMax = false, enabled = true, fail = false, unscored = false } = {}) {
    const texts = scores.map((_, index) => `candidate-${index}`);
    const metadata = texts.map((text, index) => ({ text, index, hash: hash(text) }));
    const entries = texts.map((content, uid) => ({ content, uid, world: 'test', vectorized: true }));
    const prompts = new Map();
    const activations = [];
    const requests = [];
    const imports = {
        extension_prompt_types: { IN_PROMPT: 0 },
        extension_prompt_roles: { SYSTEM: 0 },
        ModuleWorkerWrapper: class {},
        WebLlmVectorProvider: class {},
        debounce: () => () => {},
        debounce_timeout: { relaxed: 0 },
        extension_settings: { vectors: {} },
        getCurrentChatId: () => 'test-chat',
        getRequestHeaders: () => ({}),
        getStringHash: hash,
        substituteParams: text => text,
        substituteParamsExtended: (template, values) => template.replace('{{text}}', values.text),
        collapseNewlines: text => text,
        onlyUnique: (value, index, values) => values.indexOf(value) === index,
        setExtensionPrompt: (tag, text) => prompts.set(tag, text),
        getSortedEntries: async () => entries,
        event_types: { WORLDINFO_FORCE_ACTIVATE: 'world-info' },
        eventSource: { emit: async (_event, selected) => activations.push(...selected) },
    };
    const context = vm.createContext({
        console: { debug() {}, log() {}, warn() {}, error() {} },
        jQuery() {},
        $: () => ({ length: 0 }),
        toastr: { warning() {}, error() {} },
        AbortSignal,
        fetch: async (url, options) => {
            const body = JSON.parse(options.body);
            requests.push({ url, body });
            if (url.endsWith('/v1/rerank')) {
                if (fail) throw new Error('reranker unavailable');
                if (unscored) return { ok: true, json: async () => ({ results: texts.map((_, index) => ({ index })).reverse() }) };
                return { ok: true, json: async () => ({ results: scores.flatMap((score, index) =>
                    score === undefined ? [] : [{ index, relevance_score: score }]) }) };
            }
            const collection = { hashes: metadata.map(item => item.hash), metadata };
            if (url === '/api/vector/query') return { ok: true, json: async () => collection };
            if (url === '/api/vector/query-multi') {
                return { ok: true, json: async () => Object.fromEntries(body.collectionIds.map(id => [id, collection])) };
            }
            if (url === '/api/vector/list') return { ok: true, json: async () => collection.hashes };
            throw new Error(`Unexpected endpoint: ${url}`);
        },
    });
    const dependencyNames = new Map([...source.matchAll(/^import\s*\{([\s\S]*?)\}\s*from\s*'([^']+)';/gm)]
        .map(([, names, specifier]) => [specifier, names.split(',').map(name => name.trim().split(/\s+as\s+/)[0]).filter(Boolean)]));
    const module = new vm.SourceTextModule(`${source}\nexport { settings, retrieveFileChunks, injectDataBankChunks, rearrangeChat, activateWorldInfo };\nexport const readReport = () => lastRecallReport;`, { context });
    await module.link(specifier => {
        const names = dependencyNames.get(specifier);
        assert.ok(names, `Known dependency: ${specifier}`);
        return new vm.SyntheticModule(names, function () {
            for (const name of names) this.setExport(name, imports[name] ?? (() => {}));
        }, { context });
    });
    await module.evaluate();
    const api = module.namespace;
    Object.assign(api.settings, {
        rerank_enabled: enabled,
        rerank_model: 'test-model',
        rerank_score_mode: { 'test-model': mode },
        rerank_probability_threshold: threshold,
        rerank_ignore_max_entries: ignoreMax,
        chunk_count: 1,
        chunk_count_db: 1,
        insert: 1,
        max_entries: 1,
        enabled_chats: true,
        enabled_world_info: false,
        protect: 1,
        debug_enabled: false,
    });
    async function run(path) {
        let selectedTexts;
        if (path === 'file') {
            selectedTexts = (await api.retrieveFileChunks('query', 'file')).trim().split('\n').filter(Boolean);
        } else if (path === 'data_bank') {
            await api.injectDataBankChunks('query', ['file']);
            selectedTexts = (prompts.get('4_vectors_data_bank') || '').replace('Related information:\n', '').trim().split('\n').filter(Boolean);
        } else if (path === 'chat') {
            const chat = [...texts.map(mes => ({ mes, name: 'User' })), { mes: 'protected query', name: 'User' }];
            await api.rearrangeChat(chat);
            selectedTexts = (prompts.get('3_vectors') || '').replace('Past events:\n', '').trim().split('\n\n').filter(Boolean).map(text => text.replace('User: ', ''));
            assert.ok(chat.some(message => message.mes === 'protected query'), 'Protected message survives');
            assert.equal(chat.length, texts.length + 1 - selectedTexts.length, 'Only selected messages move into the prompt');
        } else {
            api.settings.enabled_world_info = true;
            await api.activateWorldInfo([{ mes: 'query' }]);
            selectedTexts = activations.map(entry => entry.content);
        }
        return { selectedTexts, report: api.readReport(), requests };
    }
    return { run };
}

for (const path of ['file', 'data_bank', 'chat', 'world_info']) {
    test(`${path}: probability threshold can reject every candidate`, async () => {
        const { run } = await loadExtension({ threshold: 0.96 });
        const { selectedTexts, report } = await run(path);
        assert.deepEqual(selectedTexts, []);
        assert.equal(report.finalCount, 0);
        assert.ok(report.candidates.every(candidate => !candidate.inserted));
    });

    test(`${path}: ignoring the count injects all probabilities at or above the threshold`, async () => {
        const { run } = await loadExtension({ ignoreMax: true });
        const { selectedTexts, report } = await run(path);
        assert.deepEqual(selectedTexts.toSorted(), ['candidate-0', 'candidate-1', 'candidate-2', 'candidate-4']);
        assert.equal(report.finalCount, 4);
        assert.equal(report.candidates.filter(candidate => candidate.inserted).length, 4);
        assert.match(report.candidates.find(candidate => candidate.text === 'candidate-3').reason, /概率/);
    });

    test(`${path}: count limit still applies after probability filtering`, async () => {
        const { run } = await loadExtension();
        const { selectedTexts, report } = await run(path);
        assert.deepEqual(selectedTexts, ['candidate-1']);
        assert.equal(report.finalCount, 1);
    });

    for (const mode of ['auto', 'logit']) {
        test(`${path}: ${mode} mode ignores the probability-only options`, async () => {
            const { run } = await loadExtension({ mode, threshold: 1, ignoreMax: true });
            assert.deepEqual((await run(path)).selectedTexts, ['candidate-1']);
        });
    }

    test(`${path}: service failure falls back to vector order with the original count limit`, async () => {
        const { run } = await loadExtension({ fail: true, ignoreMax: true, threshold: 1 });
        const { selectedTexts, report } = await run(path);
        assert.deepEqual(selectedTexts, ['candidate-0']);
        assert.equal(report.finalCount, 1);
    });

    test(`${path}: a completely unscored response falls back to vector order`, async () => {
        const { run } = await loadExtension({ unscored: true, ignoreMax: true, threshold: 1 });
        const { selectedTexts, report } = await run(path);
        assert.deepEqual(selectedTexts, ['candidate-0']);
        assert.equal(report.finalCount, 1);
    });

    test(`${path}: a single candidate is scored before applying the probability threshold`, async () => {
        const { run } = await loadExtension({ scores: [0.2] });
        const { selectedTexts, requests } = await run(path);
        assert.deepEqual(selectedTexts, []);
        assert.equal(requests.filter(request => request.url.endsWith('/v1/rerank')).length, 1);
    });

    test(`${path}: unscored and invalid probabilities cannot pass the filter`, async () => {
        const { run } = await loadExtension({ scores: [undefined, 0.8, -0.1, 1.2], ignoreMax: true });
        assert.deepEqual((await run(path)).selectedTexts, ['candidate-1']);
    });
}

test('probability filtering is inactive when the reranker is disabled', async () => {
    const { run } = await loadExtension({ enabled: false, ignoreMax: true, threshold: 1 });
    const { selectedTexts, requests } = await run('chat');
    assert.deepEqual(selectedTexts, ['candidate-0']);
    assert.equal(requests.filter(request => request.url.endsWith('/v1/rerank')).length, 0);
});
