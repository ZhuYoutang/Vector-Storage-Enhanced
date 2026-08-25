import {
    eventSource,
    event_types,
    extension_prompt_types,
    extension_prompt_roles,
    getCurrentChatId,
    getRequestHeaders,
    is_send_press,
    saveSettingsDebounced,
    setExtensionPrompt,
    substituteParams,
    generateRaw,
    substituteParamsExtended,
} from '../../../../script.js';
import {
    ModuleWorkerWrapper,
    extension_settings,
    getContext,
    modules,
    doExtrasFetch, getApiUrl,
    openThirdPartyExtensionMenu,
} from '../../../extensions.js';
import { renderTemplateAsync } from '../../../templates.js';
import { collapseNewlines, registerDebugFunction } from '../../../power-user.js';
import { SECRET_KEYS, secret_state } from '../../../secrets.js';
import { getDataBankAttachments, getDataBankAttachmentsForSource, getFileAttachment } from '../../../chats.js';
import { debounce, getStringHash as calculateHash, waitUntilCondition, onlyUnique, splitRecursive, trimToStartSentence, trimToEndSentence, escapeHtml, isTrueBoolean } from '../../../utils.js';
import { debounce_timeout } from '../../../constants.js';
import { getSortedEntries } from '../../../world-info.js';
import { textgen_types, textgenerationwebui_settings } from '../../../textgen-settings.js';
import { SlashCommandParser } from '../../../slash-commands/SlashCommandParser.js';
import { SlashCommand } from '../../../slash-commands/SlashCommand.js';
import { ARGUMENT_TYPE, SlashCommandArgument, SlashCommandNamedArgument } from '../../../slash-commands/SlashCommandArgument.js';
import { SlashCommandEnumValue, enumTypes } from '../../../slash-commands/SlashCommandEnumValue.js';
import { commonEnumProviders } from '../../../slash-commands/SlashCommandCommonEnumsProvider.js';
import { slashCommandReturnHelper } from '../../../slash-commands/SlashCommandReturnHelper.js';
import { generateWebLlmChatPrompt, isWebLlmSupported } from '../../shared.js';
import { WebLlmVectorProvider } from './webllm.js';
import { removeReasoningFromString } from '../../../reasoning.js';
import { oai_settings } from '../../../openai.js';

/**
 * @typedef {object} HashedMessage
 * @property {string} text - The hashed message text
 * @property {number} hash - The hash used as the vector key
 * @property {number} index - The index of the message in the chat
 */

export const EXTENSION_PROMPT_TAG = '3_vectors';
export const EXTENSION_PROMPT_TAG_DB = '4_vectors_data_bank';

// Force solo chunks for sources that don't support batching.
const getBatchSize = () => ['transformers', 'ollama'].includes(settings.source) ? 1 : 5;

const settings = {
    // For both
    source: 'transformers',
    alt_endpoint_url: '',
    use_alt_endpoint: false,
    include_wi: false,
    togetherai_model: 'togethercomputer/m2-bert-80M-32k-retrieval',
    openai_model: 'text-embedding-ada-002',
    electronhub_model: 'text-embedding-3-small',
    openrouter_model: 'openai/text-embedding-3-large',
    cohere_model: 'embed-english-v3.0',
    ollama_model: 'mxbai-embed-large',
    ollama_keep: false,
    vllm_model: '',
    webllm_model: '',
    google_model: 'text-embedding-005',
    chutes_model: 'chutes-qwen-qwen3-embedding-8b',
    nanogpt_model: 'text-embedding-3-small',
    siliconflow_model: 'Qwen/Qwen3-Embedding-0.6B',
    summarize: false,
    summarize_sent: false,
    summary_source: 'main',
    summary_prompt: 'Ignore previous instructions. Summarize the most important parts of the message. Limit yourself to 250 words or less. Your response should include nothing but the summary.',
    force_chunk_delimiter: '',

    // For chats
    enabled_chats: false,
    template: 'Past events:\n{{text}}',
    depth: 2,
    position: extension_prompt_types.IN_PROMPT,
    protect: 5,
    insert: 3,
    query: 2,
    message_chunk_size: 400,
    score_threshold: 0.25,

    // For files
    enabled_files: false,
    translate_files: false,
    size_threshold: 10,
    chunk_size: 5000,
    chunk_count: 2,
    overlap_percent: 0,
    only_custom_boundary: false,

    // For Data Bank
    size_threshold_db: 5,
    chunk_size_db: 2500,
    chunk_count_db: 5,
    overlap_percent_db: 0,
    file_template_db: 'Related information:\n{{text}}',
    file_position_db: extension_prompt_types.IN_PROMPT,
    file_depth_db: 4,
    file_depth_role_db: extension_prompt_roles.SYSTEM,

    // For World Info
    enabled_world_info: false,
    enabled_for_all: false,
    max_entries: 5,

    // For Reranker
    rerank_enabled: false,
    rerank_url: 'http://127.0.0.1:8080',
    rerank_key: '',
    rerank_model: 'bge-reranker-v2-m3',
    rerank_recall: 30,
    rerank_ignore_threshold: true,
    rerank_query_max_chars: 2000,
    rerank_doc_max_chars: 1600,
    rerank_timeout_ms: 60000,

    // For Diagnostics
    debug_enabled: true,
    debug_diff_threshold: false,
};

const moduleWorker = new ModuleWorkerWrapper(synchronizeChat);
const webllmProvider = new WebLlmVectorProvider();

// ##################################################################################
// ### Diagnostics: structured logging, recall reports, integrity checks          ###
// ##################################################################################

const RAG_LOG_MAX_ENTRIES = 200;
const RAG_LOG_PREVIEW_LENGTH = 160;
const LONE_SURROGATE_REGEX = /[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]/;

/**
 * Ring buffer of diagnostic log entries. Persisted (debounced) into extension settings.
 * @type {{ ts: number, level: string, type: string, message: string, data?: any }[]}
 */
const ragLogEntries = [];

/** Last recall report, also persisted so it survives page reloads. */
let lastRecallReport = null;

const saveRagLogDebounced = debounce(() => {
    if (!extension_settings.vectors) {
        return;
    }
    extension_settings.vectors.debug_log = ragLogEntries.slice(-RAG_LOG_MAX_ENTRIES);
    extension_settings.vectors.debug_last_report = lastRecallReport;
    saveSettingsDebounced();
}, 2000);

/**
 * Updates the status badge shown in the diagnostics panel.
 * @param {string} text Badge text
 * @param {string} level One of 'info', 'warn', 'error'
 */
function updateStatusBadge(text, level = 'info') {
    const badge = $('#vectors_diag_status');
    if (!badge.length) {
        return;
    }
    badge.text(text);
    badge.toggleClass('rag-badge-ok', level === 'info');
    badge.toggleClass('rag-badge-warn', level === 'warn');
    badge.toggleClass('rag-badge-err', level === 'error');
}

/**
 * Logistic function used to map raw reranker logits to a [0, 1] relevance probability.
 * llama.cpp-style backends return unnormalized logits, so percentages need this conversion.
 * @param {number} x Raw logit
 * @returns {number} Value in (0, 1)
 */
function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}

/**
 * Formats a raw reranker score for display as "raw (sigmoid%)".
 * @param {number|null|undefined} score Raw relevance score
 * @returns {string} Human readable score, '-' when unavailable
 */
function formatRerankScore(score) {
    if (!Number.isFinite(score)) {
        return '-';
    }
    return `${score.toFixed(3)} (${(sigmoid(score) * 100).toFixed(1)}%)`;
}

/**
 * Formats just the sigmoid percentage of a raw reranker score.
 * @param {number|null|undefined} score Raw relevance score
 * @returns {string} Percentage string, '-' when unavailable
 */
function formatSigmoidPercent(score) {
    if (!Number.isFinite(score)) {
        return '-';
    }
    return `${(sigmoid(score) * 100).toFixed(1)}%`;
}

/**
 * Records a structured diagnostic event.
 * @param {string} type Event type ('vectorize'|'query'|'rerank'|'inject'|'system')
 * @param {string} level 'info'|'warn'|'error'
 * @param {string} message Human readable summary
 * @param {any} [data] Optional structured payload
 */
function ragLogEvent(type, level, message, data) {
    const entry = { ts: Date.now(), level, type, message, ...(data !== undefined ? { data } : {}) };
    ragLogEntries.push(entry);
    if (ragLogEntries.length > RAG_LOG_MAX_ENTRIES) {
        ragLogEntries.splice(0, ragLogEntries.length - RAG_LOG_MAX_ENTRIES);
    }

    const consoleArgs = [`Vectors[${type}]: ${message}`];
    if (data !== undefined) {
        consoleArgs.push(data);
    }
    switch (level) {
        case 'warn':
            console.warn(...consoleArgs);
            break;
        case 'error':
            console.error(...consoleArgs);
            break;
        default:
            console.log(...consoleArgs);
            break;
    }

    renderRagLog();
    saveRagLogDebounced();
}

/**
 * Renders the log viewer and last report table in the settings panel (if present).
 */
function renderRagLog() {
    const viewer = $('#vectors_diag_log');
    if (!viewer.length) {
        return;
    }
    const filterLevel = String($('#vectors_diag_filter').val() || 'all');
    const rows = ragLogEntries
        .filter(x => filterLevel === 'all' || x.level === filterLevel)
        .slice(-100)
        .reverse()
        .map(x => {
            const time = new Date(x.ts).toLocaleTimeString();
            const detail = x.data ? `<span class="rag-log-detail">${escapeHtml(JSON.stringify(x.data))}</span>` : '';
            return `<div class="rag-log-line rag-log-${x.level}"><span class="rag-log-time">${time}</span> <span class="rag-log-type">[${escapeHtml(x.type)}]</span> ${escapeHtml(x.message)} ${detail}</div>`;
        });
    viewer.html(rows.join('') || '<div class="rag-log-line">(no log entries)</div>');
}

/**
 * Stores and renders a recall report of the most recent retrieval pass.
 * @param {object} report Recall report
 */
function recordRecallReport(report) {
    lastRecallReport = report;
    // All report-producing paths go through getRecallParams, which forces the recall
    // threshold to 0 exactly when the reranker is active and "ignore threshold" is set.
    report.thresholdIgnored = shouldRerank() && !!settings.rerank_ignore_threshold;

    const topScore = report.reranked ? (report.candidates?.[0]?.rerankScore ?? null) : null;
    const allWeak = Number.isFinite(topScore) && sigmoid(topScore) < 0.5;

    updateStatusBadge(allWeak
        ? `⚠ ${report.finalCount ?? 0} injected · all weak (top ${formatSigmoidPercent(topScore)})`
        : `✓ ${report.finalCount ?? 0} injected (${report.tookMs ?? '?'}ms)`, allWeak ? 'warn' : 'info');
    ragLogEvent(report.path, 'info', `Recall: ${report.candidates?.length ?? 0} candidates -> ${report.finalCount ?? 0} final` +
        `${report.reranked ? ' [reranked]' : ''}`, {
        topK: report.topK,
        threshold: report.threshold,
        tookMs: report.tookMs,
    });
    if (allWeak) {
        ragLogEvent(report.path, 'warn', `All reranked candidates weakly relevant (best raw ${topScore.toFixed(3)} = ${formatSigmoidPercent(topScore)}). Check recall pool/query text.`, { topScore: topScore });
    }
    renderLastReport();
    saveRagLogDebounced();
}

/**
 * Marks a failed interception in the diagnostics UI.
 * @param {string} path Retrieval path that failed
 * @param {unknown} error Error object
 */
function recordRecallError(path, error) {
    updateStatusBadge(`✗ ${path} failed`, 'error');
    ragLogEvent(path, 'error', `Retrieval failed: ${error?.message || error}`);
}

/**
 * Renders the "last recall" table into the diagnostics panel.
 */
function renderLastReport() {
    const container = $('#vectors_last_report');
    if (!container.length) {
        return;
    }
    if (!lastRecallReport) {
        container.html('<small>(no retrieval recorded yet)</small>');
        return;
    }

    const r = lastRecallReport;
    const thresholdLabel = r.thresholdIgnored
        ? `${r.threshold} <i>(forced 0 — reranker decides)</i>`
        : `${r.threshold} <i>(recall pre-filter)</i>`;
    const head = `<div class="rag-report-meta">path=<b>${escapeHtml(r.path)}</b> | topK=${r.topK} | threshold=${thresholdLabel}` +
        ` | reranked=${r.reranked ? 'yes' : 'no'} | took=${r.tookMs}ms</div>` +
        `<div class="rag-report-meta">query: ${escapeHtml(String(r.queryPreview || '').slice(0, 300))}</div>`;

    const killed = r.filteredByThreshold || [];
    const killedRows = killed.map(k =>
        `<div class="rag-report-killed">[killed by threshold] ${escapeHtml(k)}</div>`).join('');

    const topScore = r.reranked ? (r.candidates?.[0]?.rerankScore ?? null) : null;
    const weakBanner = Number.isFinite(topScore) && sigmoid(topScore) < 0.5
        ? `<div class="rag-report-killed">⚠ all candidates weakly relevant (best: ${formatRerankScore(topScore)})</div>`
        : '';

    const rows = (r.candidates || []).map(c => {
        const status = c.inserted ? '<b style="color:var(--okGreen)">INSERTED</b>' : `cut (${c.reason})`;
        return `<tr><td>${c.rank}</td><td>${escapeHtml(formatRerankScore(c.rerankScore))}</td><td>${status}</td><td class="rag-report-text">${escapeHtml(c.text)}</td></tr>`;
    }).join('');
    const table = rows
        ? `<table class="rag-report-table"><thead><tr><th>#</th><th>rerank</th><th>status</th><th>text preview</th></tr></thead><tbody>${rows}</tbody></table>`
        : '<small>(no candidates returned)</small>';

    container.html(head + weakBanner + killedRows + table);
}

/**
 * Validates chunk texts before they are vectorized. Detection only - never mutates data.
 * Flags lone surrogate pairs (broken emoji), empty chunks, tiny chunks and duplicate hashes.
 * @param {string[]} chunks Chunk texts
 * @param {string} sourceName Human readable source description for warnings
 * @returns {boolean} True if no hard problems were found
 */
function validateChunks(chunks, sourceName) {
    const problems = [];
    /** @type {Map<number, number>} */
    const seenHashes = new Map();

    chunks.forEach((rawChunk, index) => {
        const chunk = String(rawChunk);
        if (!chunk.trim()) {
            problems.push({ index, reason: 'empty chunk' });
            return;
        }
        if (LONE_SURROGATE_REGEX.test(chunk)) {
            problems.push({ index, reason: 'lone surrogate pair (broken emoji?)', preview: chunk.slice(0, 80) });
        }
        if (index > 0 && chunk.trim().length < 8) {
            problems.push({ index, reason: 'suspiciously tiny chunk', preview: chunk.slice(0, 40) });
        }
        const hash = getStringHash(chunk);
        if (seenHashes.has(hash)) {
            problems.push({ index, reason: `duplicate hash with chunk #${seenHashes.get(hash)} (identical content)` });
        } else {
            seenHashes.set(hash, index);
        }
    });

    if (problems.length > 0) {
        toastr.warning(`Found ${problems.length} suspicious chunk(s) in "${sourceName}". Details in the Diagnostics panel.`, 'Vector Storage', { timeOut: 8000 });
        for (const problem of problems) {
            ragLogEvent('vectorize', 'warn', `Chunk problem in "${sourceName}" #${problem.index}: ${problem.reason}`, problem.preview ? { preview: problem.preview } : undefined);
        }
        return false;
    }

    return true;
}

/**
 * Verifies that the number of stored hashes matches what was just ingested.
 * Catches silent insertion drops.
 * @param {string} collectionId Collection ID
 * @param {number} expectedDelta Expected number of newly stored items
 * @param {number} countBefore Saved-hash count before ingestion
 * @param {string} sourceName Source description for logs
 */
async function verifyInsertionCount(collectionId, expectedDelta, countBefore, sourceName) {
    try {
        const savedHashes = await getSavedHashes(collectionId);
        const delta = savedHashes.length - countBefore;
        if (delta !== expectedDelta) {
            toastr.warning(`Stored ${delta}/${expectedDelta} chunks for "${sourceName}". Some items may have been dropped silently!`, 'Vector Storage');
            ragLogEvent('vectorize', 'warn', `Insertion mismatch for "${sourceName}": expected ${expectedDelta}, stored ${delta}`, { collectionId });
        } else {
            ragLogEvent('vectorize', 'info', `Verified ${delta}/${expectedDelta} chunks stored for "${sourceName}"`);
        }
    } catch (error) {
        ragLogEvent('vectorize', 'warn', `Could not verify insertion for "${sourceName}"`, { error: String(error?.message || error) });
    }
}
const cachedSummaries = new Map();
const vectorApiRequiresUrl = ['llamacpp', 'vllm', 'ollama', 'koboldcpp'];

/**
 * Gets the Collection ID for a file embedded in the chat.
 * @param {string} fileUrl URL of the file
 * @returns {string} Collection ID
 */
function getFileCollectionId(fileUrl) {
    return `file_${getStringHash(fileUrl)}`;
}

async function onVectorizeAllClick() {
    try {
        if (!settings.enabled_chats) {
            return;
        }

        const chatId = getCurrentChatId();

        if (!chatId) {
            toastr.info('No chat selected', 'Vectorization aborted');
            return;
        }

        // Clear all cached summaries to ensure that new ones are created
        // upon request of a full vectorise
        cachedSummaries.clear();

        const batchSize = getBatchSize();
        const elapsedLog = [];
        let finished = false;
        $('#vectorize_progress').show();
        $('#vectorize_progress_percent').text('0');
        $('#vectorize_progress_eta').text('...');

        while (!finished) {
            if (is_send_press) {
                toastr.info('Message generation is in progress.', 'Vectorization aborted');
                throw new Error('Message generation is in progress.');
            }

            const startTime = Date.now();
            const remaining = await synchronizeChat(batchSize);
            const elapsed = Date.now() - startTime;
            elapsedLog.push(elapsed);
            finished = remaining <= 0;

            const total = getContext().chat.length;
            const processed = total - remaining;
            const processedPercent = Math.round((processed / total) * 100); // percentage of the work done
            const lastElapsed = elapsedLog.slice(-5); // last 5 elapsed times
            const averageElapsed = lastElapsed.reduce((a, b) => a + b, 0) / lastElapsed.length; // average time needed to process one item
            const pace = averageElapsed / batchSize; // time needed to process one item
            const remainingTime = Math.round(pace * remaining / 1000);

            $('#vectorize_progress_percent').text(processedPercent);
            $('#vectorize_progress_eta').text(remainingTime);

            if (chatId !== getCurrentChatId()) {
                throw new Error('Chat changed');
            }
        }
    } catch (error) {
        console.error('Vectors: Failed to vectorize all', error);
    } finally {
        $('#vectorize_progress').hide();
    }
}

let syncBlocked = false;

/**
 * Gets the chunk delimiters for splitting text.
 * @returns {string[]} Array of chunk delimiters
 */
function getChunkDelimiters() {
    const delimiters = ['\n\n', '\n', ' ', ''];

    if (settings.force_chunk_delimiter) {
        delimiters.unshift(settings.force_chunk_delimiter);
    }

    return delimiters;
}

/**
 * Splits messages into chunks before inserting them into the vector index.
 * @param {object[]} items Array of vector items
 * @returns {object[]} Array of vector items (possibly chunked)
 */
function splitByChunks(items) {
    if (settings.message_chunk_size <= 0) {
        validateChunks(items.map(x => x.text), 'chat messages');
        return items;
    }

    const chunkedItems = [];

    for (const item of items) {
        const chunks = splitRecursive(item.text, settings.message_chunk_size, getChunkDelimiters());
        validateChunks(chunks, `chat message #${item.index}`);
        for (const chunk of chunks) {
            const chunkedItem = { ...item, text: chunk };
            chunkedItems.push(chunkedItem);
        }
    }

    return chunkedItems;
}

/**
 * Summarizes messages using the Extras API method.
 * @param {HashedMessage} element hashed message
 * @returns {Promise<boolean>} Sucess
 */
async function summarizeExtra(element) {
    try {
        const url = new URL(getApiUrl());
        url.pathname = '/api/summarize';

        const apiResult = await doExtrasFetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Bypass-Tunnel-Reminder': 'bypass',
            },
            body: JSON.stringify({
                text: element.text,
                params: {},
            }),
        });

        if (apiResult.ok) {
            const data = await apiResult.json();
            element.text = data.summary;
        }
    } catch (error) {
        console.log(error);
        return false;
    }

    return true;
}

/**
 * Summarizes messages using the main API method.
 * @param {HashedMessage} element hashed message
 * @returns {Promise<boolean>} Success
 */
async function summarizeMain(element) {
    element.text = removeReasoningFromString(await generateRaw({ prompt: element.text, systemPrompt: settings.summary_prompt }));
    return true;
}

/**
 * Summarizes messages using WebLLM.
 * @param {HashedMessage} element hashed message
 * @returns {Promise<boolean>} Success
 */
async function summarizeWebLLM(element) {
    if (!isWebLlmSupported()) {
        console.warn('Vectors: WebLLM is not supported');
        return false;
    }

    const messages = [{ role: 'system', content: settings.summary_prompt }, { role: 'user', content: element.text }];
    element.text = await generateWebLlmChatPrompt(messages);

    return true;
}

/**
 * Summarizes messages using the chosen method.
 * @param {HashedMessage[]} hashedMessages Array of hashed messages
 * @param {string} endpoint Type of endpoint to use
 * @returns {Promise<HashedMessage[]>} Summarized messages
 */
async function summarize(hashedMessages, endpoint = 'main') {
    for (const element of hashedMessages) {
        const cachedSummary = cachedSummaries.get(element.hash);
        if (!cachedSummary) {
            let success = true;
            switch (endpoint) {
                case 'main':
                    success = await summarizeMain(element);
                    break;
                case 'extras':
                    success = await summarizeExtra(element);
                    break;
                case 'webllm':
                    success = await summarizeWebLLM(element);
                    break;
                default:
                    console.error('Unsupported endpoint', endpoint);
                    success = false;
                    break;
            }
            if (success) {
                cachedSummaries.set(element.hash, element.text);
            } else {
                break;
            }
        } else {
            element.text = cachedSummary;
        }
    }
    return hashedMessages;
}

async function synchronizeChat(batchSize = 5) {
    if (!settings.enabled_chats) {
        return -1;
    }

    try {
        await waitUntilCondition(() => !syncBlocked && !is_send_press, 1000);
    } catch {
        console.log('Vectors: Synchronization blocked by another process');
        return -1;
    }

    try {
        syncBlocked = true;
        const context = getContext();
        const chatId = getCurrentChatId();

        if (!chatId || !Array.isArray(context.chat)) {
            console.debug('Vectors: No chat selected');
            return -1;
        }

        const hashedMessages = context.chat.filter(x => !x.is_system).map(x => ({ text: String(substituteParams(x.mes)), hash: getStringHash(substituteParams(x.mes)), index: context.chat.indexOf(x) }));
        const hashesInCollection = await getSavedHashes(chatId);

        let newVectorItems = hashedMessages.filter(x => !hashesInCollection.includes(x.hash));
        const deletedHashes = hashesInCollection.filter(x => !hashedMessages.some(y => y.hash === x));

        if (settings.summarize) {
            newVectorItems = await summarize(newVectorItems, settings.summary_source);
        }

        if (newVectorItems.length > 0) {
            const chunkedBatch = splitByChunks(newVectorItems.slice(0, batchSize));

            console.log(`Vectors: Found ${newVectorItems.length} new items. Processing ${batchSize}...`);
            await insertVectorItems(chatId, chunkedBatch);
        }

        if (deletedHashes.length > 0) {
            await deleteVectorItems(chatId, deletedHashes);
            console.log(`Vectors: Deleted ${deletedHashes.length} old hashes`);
        }

        return newVectorItems.length - batchSize;
    } catch (error) {
        /**
         * Gets the error message for a given cause
         * @param {string} cause Error cause key
         * @returns {string} Error message
         */
        function getErrorMessage(cause) {
            switch (cause) {
                case 'api_key_missing':
                    return 'API key missing. Save it in the "API Connections" panel.';
                case 'api_url_missing':
                    return 'API URL missing. Save it in the "API Connections" panel.';
                case 'api_model_missing':
                    return 'Vectorization Source Model is required, but not set.';
                case 'extras_module_missing':
                    return 'Extras API must provide an "embeddings" module.';
                case 'webllm_not_supported':
                    return 'WebLLM extension is not installed or the model is not set.';
                default:
                    return 'Check server console for more details';
            }
        }

        console.error('Vectors: Failed to synchronize chat', error);

        const message = getErrorMessage(error.cause);
        toastr.error(message, 'Vectorization failed', { preventDuplicates: true });
        return -1;
    } finally {
        syncBlocked = false;
    }
}

/**
 * @type {Map<string, number>} Cache object for storing hash values
 */
const hashCache = new Map();

/**
 * Gets the hash value for a given string
 * @param {string} str Input string
 * @returns {number} Hash value
 */
function getStringHash(str) {
    // Check if the hash is already in the cache
    if (hashCache.has(str)) {
        return hashCache.get(str);
    }

    // Calculate the hash value
    const hash = calculateHash(str);

    // Store the hash in the cache
    hashCache.set(str, hash);

    return hash;
}

/**
 * Retrieves files from the chat and inserts them into the vector index.
 * @param {ChatMessage[]} chat Array of chat messages
 * @returns {Promise<void>}
 */
async function processFiles(chat) {
    try {
        if (!settings.enabled_files) {
            return;
        }

        const dataBankCollectionIds = await ingestDataBankAttachments();

        if (dataBankCollectionIds.length) {
            const queryText = await getQueryText(chat, 'file');
            await injectDataBankChunks(queryText, dataBankCollectionIds);
        }

        for (const message of chat) {
            // Message has no files
            if (!Array.isArray(message?.extra?.files) || !message.extra.files.length) {
                continue;
            }

            // Trim file inserted by the script
            const allFileText = String(message.mes || '').substring(0, message.extra.fileLength).trim();

            // Convert kilobytes to string length
            const thresholdLength = settings.size_threshold * 1024;

            // File is too small
            if (allFileText.length < thresholdLength) {
                continue;
            }

            message.mes = message.mes.substring(message.extra.fileLength);

            const allFileChunks = [];
            const queryText = await getQueryText(chat, 'file');

            for (const file of message.extra.files) {
                const fileName = file.name;
                const fileUrl = file.url;
                const collectionId = getFileCollectionId(fileUrl);
                const hashesInCollection = await getSavedHashes(collectionId);

                // File is not vectorized yet
                if (!hashesInCollection.length) {
                    const fileText = file.text || (await getFileAttachment(fileUrl));
                    if (!fileText) {
                        continue;
                    }
                    await vectorizeFile(fileText, fileName, collectionId, settings.chunk_size, settings.overlap_percent);
                }

                const fileChunks = await retrieveFileChunks(queryText, collectionId);
                if (fileChunks) {
                    allFileChunks.push(fileChunks);
                }
            }

            message.mes = `${allFileChunks.join('\n\n')}\n\n${message.mes}`;
        }
    } catch (error) {
        console.error('Vectors: Failed to retrieve files', error);
    }
}

/**
 * Ensures that data bank attachments are ingested and inserted into the vector index.
 * @param {string} [source] Optional source filter for data bank attachments.
 * @returns {Promise<string[]>} Collection IDs
 */
async function ingestDataBankAttachments(source) {
    // Exclude disabled files
    const dataBank = source ? getDataBankAttachmentsForSource(source, false) : getDataBankAttachments(false);
    const dataBankCollectionIds = [];

    for (const file of dataBank) {
        const collectionId = getFileCollectionId(file.url);
        const hashesInCollection = await getSavedHashes(collectionId);
        dataBankCollectionIds.push(collectionId);

        // File is already in the collection
        if (hashesInCollection.length) {
            continue;
        }

        // Download and process the file
        const fileText = await getFileAttachment(file.url);
        console.log(`Vectors: Retrieved file ${file.name} from Data Bank`);
        // Convert kilobytes to string length
        const thresholdLength = settings.size_threshold_db * 1024;
        // Use chunk size from settings if file is larger than threshold
        const chunkSize = file.size > thresholdLength ? settings.chunk_size_db : -1;
        await vectorizeFile(fileText, file.name, collectionId, chunkSize, settings.overlap_percent_db);
    }

    return dataBankCollectionIds;
}

/**
 * Inserts file chunks from the Data Bank into the prompt.
 * @param {string} queryText Text to query
 * @param {string[]} collectionIds File collection IDs
 * @returns {Promise<void>}
 */
async function injectDataBankChunks(queryText, collectionIds) {
    try {
        const { topK: recallTopK, threshold: recallThreshold } = getRecallParams(settings.chunk_count_db);
        const startedAt = Date.now();
        const queryResults = await queryMultipleCollections(collectionIds, queryText, recallTopK, recallThreshold);
        console.debug(`Vectors: Retrieved ${collectionIds.length} Data Bank collections`, queryResults);

        // Pool chunks from every collection (document order kept inside each collection)
        const pooledTexts = [];
        for (const collectionId in queryResults) {
            console.debug(`Vectors: Processing Data Bank collection ${collectionId}`, queryResults[collectionId]);
            // Keep the server's similarity order when reranking: the rerank-failure fallback
            // returns candidates in input order, which must stay relevance-ranked.
            // Document order only matters for the legacy non-rerank join.
            const collectionMetadata = queryResults[collectionId].metadata?.filter(x => x.text) || [];
            if (!shouldRerank()) {
                collectionMetadata.sort((a, b) => a.index - b.index);
            }
            const metadata = collectionMetadata.map(x => x.text).filter(onlyUnique);

            if (shouldRerank()) {
                for (const text of metadata) {
                    if (!pooledTexts.includes(text)) {
                        pooledTexts.push(text);
                    }
                }
            } else {
                pooledTexts.push(metadata.join('\n') + '\n\n');
            }
        }

        let textResult;
        let finalCandidates = [];

        if (shouldRerank()) {
            // Rerank the whole pooled candidate set, then keep only the final count
            const rerankedOrder = await rerankTexts(queryText, pooledTexts);
            finalCandidates = rerankedOrder.map((entry, rank) => ({
                rank: rank,
                rerankScore: entry.score,
                inserted: rank < settings.chunk_count_db,
                reason: rank < settings.chunk_count_db ? '' : 'final count cut',
                text: String(pooledTexts[entry.index] ?? '').slice(0, RAG_LOG_PREVIEW_LENGTH),
            }));
            textResult = rerankedOrder.slice(0, settings.chunk_count_db).map(x => pooledTexts[x.index]).join('\n') + '\n\n';
        } else {
            finalCandidates = pooledTexts.map((text, rank) => ({ rank: rank, rerankScore: null, inserted: true, reason: '', text: String(text).slice(0, RAG_LOG_PREVIEW_LENGTH) }));
            textResult = pooledTexts.join('');
        }

        recordRecallReport({
            ts: Date.now(),
            path: 'data_bank',
            queryPreview: queryText,
            topK: recallTopK,
            threshold: recallThreshold ?? settings.score_threshold,
            reranked: shouldRerank(),
            tookMs: Date.now() - startedAt,
            finalCount: finalCandidates.filter(x => x.inserted).length,
            filteredByThreshold: [],
            candidates: finalCandidates,
        });

        if (!textResult || !textResult.trim()) {
            console.debug('Vectors: No Data Bank chunks found');
            return;
        }

        const insertedText = substituteParamsExtended(settings.file_template_db, { text: textResult });
        setExtensionPrompt(EXTENSION_PROMPT_TAG_DB, insertedText, settings.file_position_db, settings.file_depth_db, settings.include_wi, settings.file_depth_role_db);
    } catch (error) {
        recordRecallError('data_bank', error);
        console.error('Vectors: Failed to insert Data Bank chunks', error);
    }
}

/**
 * Retrieves file chunks from the vector index and inserts them into the chat.
 * @param {string} queryText Text to query
 * @param {string} collectionId File collection ID
 * @returns {Promise<string>} Retrieved file text
 */
async function retrieveFileChunks(queryText, collectionId) {
    console.debug(`Vectors: Retrieving file chunks for collection ${collectionId}`, queryText);
    const { topK: recallTopK, threshold: recallThreshold } = getRecallParams(settings.chunk_count);
    const startedAt = Date.now();
    const queryResults = await queryCollection(collectionId, queryText, recallTopK, recallThreshold);
    console.debug(`Vectors: Retrieved ${queryResults.hashes.length} file chunks for collection ${collectionId}`, queryResults);
    // Same as the Data Bank path: similarity order must survive until rerankTexts so a
    // fallback (reranker down) still yields relevance-ranked results.
    const fileMetadata = queryResults.metadata.filter(x => x.text);
    if (!shouldRerank()) {
        fileMetadata.sort((a, b) => a.index - b.index);
    }
    const metadata = fileMetadata.map(x => x.text).filter(onlyUnique);

    let fileText;
    let finalCandidates;

    if (shouldRerank()) {
        const rerankedOrder = await rerankTexts(queryText, metadata);
        finalCandidates = rerankedOrder.map((entry, rank) => ({
            rank: rank,
            rerankScore: entry.score,
            inserted: rank < settings.chunk_count,
            reason: rank < settings.chunk_count ? '' : 'final count cut',
            text: String(metadata[entry.index] ?? '').slice(0, RAG_LOG_PREVIEW_LENGTH),
        }));
        fileText = rerankedOrder.slice(0, settings.chunk_count).map(x => metadata[x.index]).join('\n');
    } else {
        finalCandidates = metadata.map((text, rank) => ({ rank: rank, rerankScore: null, inserted: true, reason: '', text: String(text).slice(0, RAG_LOG_PREVIEW_LENGTH) }));
        fileText = metadata.join('\n');
    }

    recordRecallReport({
        ts: Date.now(),
        path: 'file',
        queryPreview: queryText,
        topK: recallTopK,
        threshold: recallThreshold ?? settings.score_threshold,
        reranked: shouldRerank(),
        tookMs: Date.now() - startedAt,
        finalCount: finalCandidates.filter(x => x.inserted).length,
        filteredByThreshold: [],
        candidates: finalCandidates,
    });

    return fileText;
}

/**
 * Vectorizes a file and inserts it into the vector index.
 * @param {string} fileText File text
 * @param {string} fileName File name
 * @param {string} collectionId File collection ID
 * @param {number} chunkSize Chunk size
 * @param {number} overlapPercent Overlap size (in %)
 * @returns {Promise<boolean>} True if successful, false if not
 */
async function vectorizeFile(fileText, fileName, collectionId, chunkSize, overlapPercent) {
    let toast = jQuery();

    try {
        if (settings.translate_files && typeof globalThis.translate === 'function') {
            console.log(`Vectors: Translating file ${fileName} to English...`);
            const translatedText = await globalThis.translate(fileText, 'en');
            fileText = translatedText;
        }

        const batchSize = getBatchSize();
        const toastBody = $('<span>').text('This may take a while. Please wait...');
        toast = toastr.info(toastBody, `Ingesting file ${escapeHtml(fileName)}`, { closeButton: false, escapeHtml: false, timeOut: 0, extendedTimeOut: 0 });
        const overlapSize = Math.round(chunkSize * overlapPercent / 100);
        const delimiters = getChunkDelimiters();
        // Overlap should not be included in chunk size. It will be later compensated by overlapChunks
        chunkSize = overlapSize > 0 ? (chunkSize - overlapSize) : chunkSize;
        const applyOverlap = (x, y, z) => overlapSize > 0 ? overlapChunks(x, y, z, overlapSize) : x;
        const chunks = settings.only_custom_boundary && settings.force_chunk_delimiter
            ? fileText.split(settings.force_chunk_delimiter).map(applyOverlap)
            : splitRecursive(fileText, chunkSize, delimiters).map(applyOverlap);
        console.debug(`Vectors: Split file ${fileName} into ${chunks.length} chunks with ${overlapPercent}% overlap`, chunks);

        // Integrity check: broken emoji (lone surrogates), empty/tiny/duplicate chunks.
        // Detection only - the data is left untouched, cleaning stays upstream.
        validateChunks(chunks, fileName);
        ragLogEvent('vectorize', 'info', `Vectorizing "${fileName}": ${chunks.length} chunks`, { collectionId, overlapPercent, chunkSize });

        const items = chunks.map((chunk, index) => ({ hash: getStringHash(chunk), text: chunk, index: index }));

        let savedHashesBefore = 0;
        try {
            savedHashesBefore = (await getSavedHashes(collectionId)).length;
        } catch {
            // Non-fatal: collection may not exist yet
        }

        for (let i = 0; i < items.length; i += batchSize) {
            toastBody.text(`${i}/${items.length} (${Math.round((i / items.length) * 100)}%) chunks processed`);
            const chunkedBatch = items.slice(i, i + batchSize);
            await insertVectorItems(collectionId, chunkedBatch);
        }

        // Post-insert verification: catches silent insertion drops
        await verifyInsertionCount(collectionId, items.length, savedHashesBefore, fileName);

        toastr.clear(toast);
        console.log(`Vectors: Inserted ${chunks.length} vector items for file ${fileName} into ${collectionId}`);
        return true;
    } catch (error) {
        toastr.clear(toast);
        toastr.error(String(error), 'Failed to vectorize file', { preventDuplicates: true });
        console.error('Vectors: Failed to vectorize file', error);
        return false;
    }
}

/**
 * Removes the most relevant messages from the chat and displays them in the extension prompt
 * @param {ChatMessage[]} chat Array of chat messages
 * @param {number} _contextSize Context size (unused)
 * @param {function} _abort Abort function (unused)
 * @param {string} type Generation type
 */
async function rearrangeChat(chat, _contextSize, _abort, type) {
    try {
        if (type === 'quiet') {
            console.debug('Vectors: Skipping quiet prompt');
            return;
        }

        // Clear the extension prompt
        setExtensionPrompt(EXTENSION_PROMPT_TAG, '', settings.position, settings.depth, settings.include_wi);
        setExtensionPrompt(EXTENSION_PROMPT_TAG_DB, '', settings.file_position_db, settings.file_depth_db, settings.include_wi, settings.file_depth_role_db);

        if (settings.enabled_files) {
            await processFiles(chat);
        }

        if (settings.enabled_world_info) {
            await activateWorldInfo(chat);
        }

        if (!settings.enabled_chats) {
            return;
        }

        const chatId = getCurrentChatId();

        if (!chatId || !Array.isArray(chat)) {
            console.debug('Vectors: No chat selected');
            return;
        }

        if (chat.length < settings.protect) {
            console.debug(`Vectors: Not enough messages to rearrange (less than ${settings.protect})`);
            return;
        }

        const queryText = await getQueryText(chat, 'chat');

        if (queryText.length === 0) {
            console.debug('Vectors: No text to query');
            return;
        }

        // Wide-recall parameters: over-fetch and optionally ignore the embedding threshold,
        // letting the reranker make the final decision ("wide entry, strict exit").
        const { topK: recallTopK, threshold: recallThreshold } = getRecallParams(settings.insert);
        const startedAt = Date.now();

        // Get the most relevant messages, excluding the last few
        const queryResults = await queryCollection(chatId, queryText, recallTopK, recallThreshold);

        // Fix: derive hashes from metadata (which IS filtered server-side by score threshold),
        // otherwise the score threshold silently has no effect on the chat path.
        // queryResults.metadata arrives sorted by similarity desc; candidate order must follow
        // it so a rerank-failure fallback keeps vector ranking instead of chat chronology.
        const queryHashes = (queryResults.metadata || []).map(x => Number(x.hash)).filter(onlyUnique);
        const retainMessages = chat.slice(-settings.protect);

        // Map each hash to its first matching message in chat order (protected messages skipped)
        const messageByHash = new Map();
        for (const message of chat) {
            if (retainMessages.includes(message) || !message.mes) {
                continue;
            }
            const hash = getStringHash(substituteParams(message.mes));
            if (!messageByHash.has(hash)) {
                messageByHash.set(hash, message);
            }
        }

        // Collect candidate messages in vector-similarity order
        const candidateMessages = [];
        for (const hash of queryHashes) {
            const message = messageByHash.get(hash);
            if (message) {
                candidateMessages.push({ message: message, hash: hash });
            }
        }

        // Optional diagnostics: figure out what the score threshold has killed
        let killedByThreshold = [];
        if (settings.debug_enabled && settings.debug_diff_threshold && recallThreshold !== 0) {
            try {
                const unfilteredResults = await queryCollection(chatId, queryText, recallTopK, 0);
                const unfilteredHashes = new Set((unfilteredResults.metadata || []).map(x => Number(x.hash)));
                const seenKilledHashes = new Set();

                for (const message of chat) {
                    if (retainMessages.includes(message) || !message.mes) {
                        continue;
                    }
                    const hash = getStringHash(substituteParams(message.mes));
                    // Present in the unfiltered result set, but rejected by the score threshold
                    if (unfilteredHashes.has(hash) && !queryHashes.includes(hash) && !seenKilledHashes.has(hash)) {
                        seenKilledHashes.add(hash);
                        killedByThreshold.push(collapseNewlines(String(message.mes)).trim().slice(0, RAG_LOG_PREVIEW_LENGTH));
                    }
                }
                ragLogEvent('query', 'info', `Threshold debug: ${killedByThreshold.length} candidate(s) killed by score threshold`, { count: killedByThreshold.length });
            } catch (error) {
                ragLogEvent('query', 'warn', 'Threshold diff query failed', { error: String(error?.message || error) });
            }
        }

        // Rerank candidates against the query, then keep only the final count
        const rerankedOrder = await rerankTexts(queryText, candidateMessages.map(c => collapseNewlines(String(c.message.mes)).trim()));
        const rankedCandidates = rerankedOrder.map((entry, rank) => ({
            ...candidateMessages[entry.index],
            rerankScore: entry.score,
            rank: rank,
        }));
        const finalCandidates = rankedCandidates.slice(0, settings.insert);
        const insertedHashes = new Set(finalCandidates.map(c => c.hash));

        const queriedMessages = [];
        for (const message of chat) {
            if (!message.mes) {
                continue;
            }
            const hash = getStringHash(substituteParams(message.mes));
            if (insertedHashes.has(hash) && !queriedMessages.includes(message)) {
                queriedMessages.push(message);
            }
        }

        // Rearrange queried messages to match relevance order
        // Order is reversed because more relevant are at the lower ranks
        queriedMessages.sort((a, b) => {
            const rankA = rankedCandidates.find(x => x.message === a)?.rank ?? 0;
            const rankB = rankedCandidates.find(x => x.message === b)?.rank ?? 0;
            return rankB - rankA;
        });

        recordRecallReport({
            ts: Date.now(),
            path: 'chat',
            queryPreview: queryText,
            topK: recallTopK,
            threshold: recallThreshold ?? settings.score_threshold,
            reranked: shouldRerank(),
            tookMs: Date.now() - startedAt,
            finalCount: queriedMessages.length,
            filteredByThreshold: killedByThreshold,
            candidates: rankedCandidates.map(c => ({
                rank: c.rank,
                rerankScore: c.rerankScore,
                inserted: insertedHashes.has(c.hash),
                reason: insertedHashes.has(c.hash) ? '' : 'final count cut',
                text: collapseNewlines(String(c.message.mes)).trim().slice(0, RAG_LOG_PREVIEW_LENGTH),
            })),
        });

        // Remove queried messages from the original chat array
        for (const message of chat) {
            if (queriedMessages.includes(message)) {
                chat.splice(chat.indexOf(message), 1);
            }
        }

        if (queriedMessages.length === 0) {
            console.debug('Vectors: No relevant messages found');
            return;
        }

        // Format queried messages into a single string
        const insertedText = getPromptText(queriedMessages);
        setExtensionPrompt(EXTENSION_PROMPT_TAG, insertedText, settings.position, settings.depth, settings.include_wi);
    } catch (error) {
        recordRecallError('chat', error);
        toastr.error('Generation interceptor aborted. Check browser console for more details.', 'Vector Storage');
        console.error('Vectors: Failed to rearrange chat', error);
    }
}

/**
 * @param {any[]} queriedMessages
 * @returns {string}
 */
function getPromptText(queriedMessages) {
    const queriedText = queriedMessages.map(x => collapseNewlines(`${x.name}: ${x.mes}`).trim()).join('\n\n');
    console.log('Vectors: relevant past messages found.\n', queriedText);
    return substituteParamsExtended(settings.template, { text: queriedText });
}

/**
 * Modifies text chunks to include overlap with adjacent chunks.
 * @param {string} chunk Current item
 * @param {number} index Current index
 * @param {string[]} chunks List of chunks
 * @param {number} overlapSize Size of the overlap
 * @returns {string} Overlapped chunks, with overlap trimmed to sentence boundaries
 */
function overlapChunks(chunk, index, chunks, overlapSize) {
    const halfOverlap = Math.floor(overlapSize / 2);
    const nextChunk = chunks[index + 1];
    const prevChunk = chunks[index - 1];

    const nextOverlap = trimToEndSentence(nextChunk?.substring(0, halfOverlap)) || '';
    const prevOverlap = trimToStartSentence(prevChunk?.substring(prevChunk.length - halfOverlap)) || '';
    const overlappedChunk = [prevOverlap, chunk, nextOverlap].filter(x => x).join(' ');

    return overlappedChunk;
}

globalThis.vectors_rearrangeChat_thirdparty = rearrangeChat;

const onChatEvent = debounce(async () => await moduleWorker.update(), debounce_timeout.relaxed);

/**
 * Gets the text to query from the chat
 * @param {ChatMessage[]} chat Chat messages
 * @param {'file'|'chat'|'world-info'} initiator Initiator of the query
 * @returns {Promise<string>} Text to query
 */
async function getQueryText(chat, initiator) {
    const getTextWithoutAttachments = (x) => {
        const fileLength = x?.extra?.fileLength || 0;
        return String(x?.mes || '').substring(fileLength).trim();
    };

    let hashedMessages = chat
        .map(x => ({ text: substituteParams(getTextWithoutAttachments(x)), hash: getStringHash(substituteParams(getTextWithoutAttachments(x))), index: chat.indexOf(x) }))
        .filter(x => x.text)
        .reverse()
        .slice(0, settings.query);

    if (initiator === 'chat' && settings.enabled_chats && settings.summarize && settings.summarize_sent) {
        hashedMessages = await summarize(hashedMessages, settings.summary_source);
    }

    const queryText = hashedMessages.map(x => x.text).join('\n');

    return collapseNewlines(queryText).trim();
}

/**
 * Gets common body parameters for vector requests.
 * @param {object} args Additional arguments
 * @returns {object} Request body
 */
function getVectorsRequestBody(args = {}) {
    const body = Object.assign({}, args);
    switch (settings.source) {
        case 'extras':
            body.extrasUrl = extension_settings.apiUrl;
            body.extrasKey = extension_settings.apiKey;
            break;
        case 'electronhub':
            body.model = extension_settings.vectors.electronhub_model;
            break;
        case 'openrouter':
            body.model = extension_settings.vectors.openrouter_model;
            break;
        case 'togetherai':
            body.model = extension_settings.vectors.togetherai_model;
            break;
        case 'openai':
            body.model = extension_settings.vectors.openai_model;
            break;
        case 'cohere':
            body.model = extension_settings.vectors.cohere_model;
            break;
        case 'ollama':
            body.model = extension_settings.vectors.ollama_model;
            body.apiUrl = settings.use_alt_endpoint ? settings.alt_endpoint_url : textgenerationwebui_settings.server_urls[textgen_types.OLLAMA];
            body.keep = !!extension_settings.vectors.ollama_keep;
            break;
        case 'llamacpp':
            body.apiUrl = settings.use_alt_endpoint ? settings.alt_endpoint_url : textgenerationwebui_settings.server_urls[textgen_types.LLAMACPP];
            break;
        case 'vllm':
            body.apiUrl = settings.use_alt_endpoint ? settings.alt_endpoint_url : textgenerationwebui_settings.server_urls[textgen_types.VLLM];
            body.model = extension_settings.vectors.vllm_model;
            break;
        case 'webllm':
            body.model = extension_settings.vectors.webllm_model;
            break;
        case 'palm':
            body.model = extension_settings.vectors.google_model;
            body.api = 'makersuite';
            break;
        case 'vertexai':
            body.model = extension_settings.vectors.google_model;
            body.api = 'vertexai';
            body.vertexai_auth_mode = oai_settings.vertexai_auth_mode;
            body.vertexai_region = oai_settings.vertexai_region;
            body.vertexai_express_project_id = oai_settings.vertexai_express_project_id;
            break;
        case 'chutes':
            body.model = extension_settings.vectors.chutes_model;
            break;
        case 'nanogpt':
            body.model = extension_settings.vectors.nanogpt_model;
            break;
        case 'siliconflow':
            body.model = extension_settings.vectors.siliconflow_model;
            body.siliconflow_endpoint = oai_settings.siliconflow_endpoint;
            break;
        default:
            break;
    }
    return body;
}

/**
 * Gets additional arguments for vector requests.
 * @param {string[]} items Items to embed
 * @returns {Promise<object>} Additional arguments
 */
async function getAdditionalArgs(items) {
    const args = {};
    switch (settings.source) {
        case 'webllm':
            args.embeddings = await createWebLlmEmbeddings(items);
            break;
        case 'koboldcpp': {
            const { embeddings, model } = await createKoboldCppEmbeddings(items);
            args.embeddings = embeddings;
            args.model = model;
            break;
        }
    }
    return args;
}

/**
 * Gets the saved hashes for a collection
* @param {string} collectionId
* @returns {Promise<number[]>} Saved hashes
*/
async function getSavedHashes(collectionId) {
    const args = await getAdditionalArgs([]);
    const response = await fetch('/api/vector/list', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            ...getVectorsRequestBody(args),
            collectionId: collectionId,
            source: settings.source,
        }),
    });

    if (!response.ok) {
        throw new Error(`Failed to get saved hashes for collection ${collectionId}`);
    }

    const hashes = await response.json();
    return hashes;
}

/**
 * Inserts vector items into a collection
 * @param {string} collectionId - The collection to insert into
 * @param {{ hash: number, text: string }[]} items - The items to insert
 * @returns {Promise<void>}
 */
async function insertVectorItems(collectionId, items) {
    throwIfSourceInvalid();

    const args = await getAdditionalArgs(items.map(x => x.text));
    const response = await fetch('/api/vector/insert', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            ...getVectorsRequestBody(args),
            collectionId: collectionId,
            items: items,
            source: settings.source,
        }),
    });

    if (!response.ok) {
        throw new Error(`Failed to insert vector items for collection ${collectionId}`);
    }
}

/**
 * Throws an error if the source is invalid (missing API key or URL, or missing module)
 */
function throwIfSourceInvalid() {
    if (settings.source === 'openai' && !secret_state[SECRET_KEYS.OPENAI] ||
        settings.source === 'electronhub' && !secret_state[SECRET_KEYS.ELECTRONHUB] ||
        settings.source === 'chutes' && !secret_state[SECRET_KEYS.CHUTES] ||
        settings.source === 'nanogpt' && !secret_state[SECRET_KEYS.NANOGPT] ||
        settings.source === 'openrouter' && !secret_state[SECRET_KEYS.OPENROUTER] ||
        settings.source === 'palm' && !secret_state[SECRET_KEYS.MAKERSUITE] ||
        settings.source === 'vertexai' && !secret_state[SECRET_KEYS.VERTEXAI] && !secret_state[SECRET_KEYS.VERTEXAI_SERVICE_ACCOUNT] ||
        settings.source === 'mistral' && !secret_state[SECRET_KEYS.MISTRALAI] ||
        settings.source === 'togetherai' && !secret_state[SECRET_KEYS.TOGETHERAI] ||
        settings.source === 'nomicai' && !secret_state[SECRET_KEYS.NOMICAI] ||
        settings.source === 'cohere' && !secret_state[SECRET_KEYS.COHERE] ||
        settings.source === 'siliconflow' && !secret_state[SECRET_KEYS.SILICONFLOW]) {
        throw new Error('Vectors: API key missing', { cause: 'api_key_missing' });
    }

    if (vectorApiRequiresUrl.includes(settings.source) && settings.use_alt_endpoint) {
        if (!settings.alt_endpoint_url) {
            throw new Error('Vectors: API URL missing', { cause: 'api_url_missing' });
        }
    } else {
        if (settings.source === 'ollama' && !textgenerationwebui_settings.server_urls[textgen_types.OLLAMA] ||
            settings.source === 'vllm' && !textgenerationwebui_settings.server_urls[textgen_types.VLLM] ||
            settings.source === 'koboldcpp' && !textgenerationwebui_settings.server_urls[textgen_types.KOBOLDCPP] ||
            settings.source === 'llamacpp' && !textgenerationwebui_settings.server_urls[textgen_types.LLAMACPP]) {
            throw new Error('Vectors: API URL missing', { cause: 'api_url_missing' });
        }
    }

    if (settings.source === 'ollama' && !settings.ollama_model || settings.source === 'vllm' && !settings.vllm_model) {
        throw new Error('Vectors: API model missing', { cause: 'api_model_missing' });
    }

    if (settings.source === 'extras' && !modules.includes('embeddings')) {
        throw new Error('Vectors: Embeddings module missing', { cause: 'extras_module_missing' });
    }

    if (settings.source === 'webllm' && (!isWebLlmSupported() || !settings.webllm_model)) {
        throw new Error('Vectors: WebLLM is not supported', { cause: 'webllm_not_supported' });
    }
}

/**
 * Deletes vector items from a collection
 * @param {string} collectionId - The collection to delete from
 * @param {number[]} hashes - The hashes of the items to delete
 * @returns {Promise<void>}
 */
async function deleteVectorItems(collectionId, hashes) {
    const response = await fetch('/api/vector/delete', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            ...getVectorsRequestBody(),
            collectionId: collectionId,
            hashes: hashes,
            source: settings.source,
        }),
    });

    if (!response.ok) {
        throw new Error(`Failed to delete vector items for collection ${collectionId}`);
    }
}

/**
 * @param {string} collectionId - The collection to query
 * @param {string} searchText - The text to query
 * @param {number} topK - The number of results to return
 * @param {number} [thresholdOverride] - Optional score threshold override
 * @returns {Promise<{ hashes: number[], metadata: object[]}>} - Hashes of the results
 */
async function queryCollection(collectionId, searchText, topK, thresholdOverride) {
    const args = await getAdditionalArgs([searchText]);
    const response = await fetch('/api/vector/query', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            ...getVectorsRequestBody(args),
            collectionId: collectionId,
            searchText: searchText,
            topK: topK,
            source: settings.source,
            threshold: thresholdOverride ?? settings.score_threshold,
        }),
    });

    if (!response.ok) {
        throw new Error(`Failed to query collection ${collectionId}`);
    }

    return await response.json();
}

/**
 * Queries multiple collections for a given text.
 * @param {string[]} collectionIds - Collection IDs to query
 * @param {string} searchText - Text to query
 * @param {number} topK - Number of results to return
 * @param {number} threshold - Score threshold
 * @returns {Promise<Record<string, { hashes: number[], metadata: object[] }>>} - Results mapped to collection IDs
 */
async function queryMultipleCollections(collectionIds, searchText, topK, threshold) {
    const args = await getAdditionalArgs([searchText]);
    const response = await fetch('/api/vector/query-multi', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            ...getVectorsRequestBody(args),
            collectionIds: collectionIds,
            searchText: searchText,
            topK: topK,
            source: settings.source,
            threshold: threshold ?? settings.score_threshold,
        }),
    });

    if (!response.ok) {
        throw new Error('Failed to query multiple collections');
    }

    return await response.json();
}

/**
 * Returns true when the reranker should be applied to retrieval results.
 * @returns {boolean}
 */
function shouldRerank() {
    return !!settings.rerank_enabled;
}

/**
 * Gets the recall parameters for a retrieval path when the reranker is active.
 * @param {number} finalCount Final number of results wanted after reranking
 * @returns {{ topK: number, threshold?: number }} Recall query parameters
 */
function getRecallParams(finalCount) {
    if (!shouldRerank()) {
        return { topK: finalCount };
    }
    const recall = Math.max(Number(settings.rerank_recall) || 30, finalCount);
    const threshold = settings.rerank_ignore_threshold ? 0 : undefined;
    return { topK: recall, threshold };
}

let rerankWarnedOnce = false;

/**
 * Reranks candidate texts against a query using an external /v1/rerank endpoint
 * (llama.cpp server style; also compatible with TEI/Jina-style responses).
 * Falls back to the original order on any failure.
 * @param {string} queryText Query text
 * @param {string[]} texts Candidate texts (index-mapped)
 * @param {boolean} [force] Bypass the global enable switch (used by the connection test)
 * @returns {Promise<{ index: number, score: number | null }[]>} Candidates ordered by relevance
 */
async function rerankTexts(queryText, texts, force = false) {
    const originalOrder = texts.map((_, index) => ({ index, score: null }));
    if ((!shouldRerank() && !force) || texts.length <= 1) {
        return originalOrder;
    }

    const baseUrl = String(settings.rerank_url || '').trim().replace(/\/+$/, '');
    if (!baseUrl) {
        return originalOrder;
    }

    const maxChars = Number(settings.rerank_query_max_chars);
    const trimmedQuery = Number.isFinite(maxChars) && maxChars > 0 && queryText.length > maxChars
        ? `${queryText.slice(0, maxChars)}…`
        : queryText;

    // Bound the document side too: chat-path candidates are full messages and can be huge,
    // which is the main driver of CPU rerank latency.
    const docMaxChars = Number(settings.rerank_doc_max_chars);
    const boundedDocs = texts.map(text => {
        const str = String(text);
        return Number.isFinite(docMaxChars) && docMaxChars > 0 && str.length > docMaxChars
            ? `${str.slice(0, docMaxChars)}…`
            : str;
    });

    /** @type {Record<string, string>} */
    const headers = { 'Content-Type': 'application/json' };
    const apiKey = String(settings.rerank_key || '').trim();
    if (apiKey) {
        headers['Authorization'] = `Bearer ${apiKey}`;
    }

    const timeoutMs = Number(settings.rerank_timeout_ms);
    const signal = Number.isFinite(timeoutMs) && timeoutMs > 0 && typeof AbortSignal !== 'undefined' && AbortSignal.timeout
        ? AbortSignal.timeout(timeoutMs)
        : undefined;

    const startedAt = Date.now();
    const totalChars = trimmedQuery.length + boundedDocs.reduce((sum, doc) => sum + doc.length, 0);
    ragLogEvent('rerank', 'info', `Dispatching ${boundedDocs.length} pairs (~${Math.round(totalChars / 1000)}k chars total)`, { timeoutMs: timeoutMs || 'none' });

    try {
        const response = await fetch(`${baseUrl}/v1/rerank`, {
            method: 'POST',
            headers: headers,
            signal: signal,
            body: JSON.stringify({
                model: String(settings.rerank_model || '').trim(),
                query: trimmedQuery,
                documents: boundedDocs,
                top_n: texts.length,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        const results = Array.isArray(data?.results) ? data.results : [];
        if (!results.length) {
            throw new Error('empty results');
        }

        // Treat missing/garbage scores as null instead of 0: llama.cpp-style backends return
        // raw logits where negative values are common, and a fake 0 would outrank real negatives.
        const scored = results
            .map(x => ({ index: Number(x.index), score: Number(x.relevance_score ?? x.score) }))
            .filter(x => Number.isInteger(x.index) && x.index >= 0 && x.index < texts.length)
            .map(x => ({ index: x.index, score: Number.isFinite(x.score) ? x.score : null }));

        // Append anything the API skipped, preserving the original order at the end
        const seen = new Set(scored.map(x => x.index));
        for (let index = 0; index < texts.length; index++) {
            if (!seen.has(index)) {
                scored.push({ index: index, score: null });
            }
        }

        // Defensive: not every /v1/rerank backend returns results sorted by relevance.
        // Enforce descending score order; ties keep original index order, null scores sink.
        scored.sort((a, b) => {
            if (a.score === null || b.score === null) {
                return (a.score === null ? 1 : 0) - (b.score === null ? 1 : 0);
            }
            return b.score - a.score || a.index - b.index;
        });

        ragLogEvent('rerank', 'info', `Reranked ${texts.length} docs in ${Date.now() - startedAt}ms`, {
            topScore: scored[0]?.score,
            topSigmoidPct: formatSigmoidPercent(scored[0]?.score),
        });
        rerankWarnedOnce = false;
        return scored;
    } catch (error) {
        const isTimeout = error?.name === 'AbortError' || error?.name === 'TimeoutError';
        const reason = isTimeout
            ? `timed out after ${timeoutMs}ms (lower Recall count / truncations, or raise the timeout)`
            : String(error?.message || error);
        console.error('Vectors: Reranker failed, falling back to vector order', error);
        ragLogEvent('rerank', 'warn', `Reranker failed: ${reason}. Falling back to vector order.`);
        if (!rerankWarnedOnce) {
            toastr.warning(`Reranker failed (${reason}). Using embedding order instead.`, 'Vector Storage');
            rerankWarnedOnce = true;
        }
        return originalOrder;
    }
}

/**
 * Runs a quick connectivity test against the configured reranker endpoint.
 * @returns {Promise<{ ok: boolean, message: string, ms: number }>}
 */
async function testRerankerConnection() {
    const startedAt = Date.now();
    try {
        const ordered = await rerankTexts('What is the capital of France?', [
            'The capital of France is Paris.',
            'Bananas are rich in potassium.',
            'The Eiffel Tower is located in Paris.',
        ], true);
        const ms = Date.now() - startedAt;
        const topText = ordered[0]?.score !== null ? `top score ${ordered[0].score.toFixed(3)}` : 'no scores returned';
        return { ok: ordered[0]?.score !== null, message: `Reranker responded in ${ms}ms (${topText})`, ms };
    } catch (error) {
        return { ok: false, message: `Reranker test failed: ${error?.message || error}`, ms: Date.now() - startedAt };
    }
}

/**
 * Purges the vector index for a file.
 * @param {string} fileUrl File URL to purge
 */
async function purgeFileVectorIndex(fileUrl) {
    try {
        if (!settings.enabled_files) {
            return;
        }

        console.log(`Vectors: Purging file vector index for ${fileUrl}`);
        const collectionId = getFileCollectionId(fileUrl);

        const response = await fetch('/api/vector/purge', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                ...getVectorsRequestBody(),
                collectionId: collectionId,
            }),
        });

        if (!response.ok) {
            throw new Error(`Could not delete vector index for collection ${collectionId}`);
        }

        console.log(`Vectors: Purged vector index for collection ${collectionId}`);
    } catch (error) {
        console.error('Vectors: Failed to purge file', error);
    }
}

/**
 * Purges the vector index for a collection.
 * @param {string} collectionId Collection ID to purge
 * @returns <Promise<boolean>> True if deleted, false if not
 */
async function purgeVectorIndex(collectionId) {
    try {
        if (!settings.enabled_chats) {
            return true;
        }

        const response = await fetch('/api/vector/purge', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                ...getVectorsRequestBody(),
                collectionId: collectionId,
            }),
        });

        if (!response.ok) {
            throw new Error(`Could not delete vector index for collection ${collectionId}`);
        }

        console.log(`Vectors: Purged vector index for collection ${collectionId}`);
        return true;
    } catch (error) {
        console.error('Vectors: Failed to purge', error);
        return false;
    }
}

/**
 * Purges all vector indexes.
 */
async function purgeAllVectorIndexes() {
    try {
        const response = await fetch('/api/vector/purge-all', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                ...getVectorsRequestBody(),
            }),
        });

        if (!response.ok) {
            throw new Error('Failed to purge all vector indexes');
        }

        console.log('Vectors: Purged all vector indexes');
        toastr.success('All vector indexes purged', 'Purge successful');
    } catch (error) {
        console.error('Vectors: Failed to purge all', error);
        toastr.error('Failed to purge all vector indexes', 'Purge failed');
    }
}

function toggleSettings() {
    $('#vectors_files_settings').toggle(!!settings.enabled_files);
    $('#vectors_chats_settings').toggle(!!settings.enabled_chats);
    $('#vectors_world_info_settings').toggle(!!settings.enabled_world_info);
    $('#vectors_rerank_settings').toggle(!!settings.rerank_enabled);
    $('#together_vectorsModel').toggle(settings.source === 'togetherai');
    $('#openai_vectorsModel').toggle(settings.source === 'openai');
    $('#electronhub_vectorsModel').toggle(settings.source === 'electronhub');
    $('#chutes_vectorsModel').toggle(settings.source === 'chutes');
    $('#nanogpt_vectorsModel').toggle(settings.source === 'nanogpt');
    $('#openrouter_vectorsModel').toggle(settings.source === 'openrouter');
    $('#cohere_vectorsModel').toggle(settings.source === 'cohere');
    $('#ollama_vectorsModel').toggle(settings.source === 'ollama');
    $('#llamacpp_vectorsModel').toggle(settings.source === 'llamacpp');
    $('#vllm_vectorsModel').toggle(settings.source === 'vllm');
    $('#nomicai_apiKey').toggle(settings.source === 'nomicai');
    $('#webllm_vectorsModel').toggle(settings.source === 'webllm');
    $('#koboldcpp_vectorsModel').toggle(settings.source === 'koboldcpp');
    $('#google_vectorsModel').toggle(settings.source === 'palm' || settings.source === 'vertexai');
    $('#siliconflow_vectorsModel').toggle(settings.source === 'siliconflow');
    $('#vector_altEndpointUrl').toggle(vectorApiRequiresUrl.includes(settings.source));
    switch (settings.source) {
        case 'webllm':
            loadWebLlmModels();
            break;
        case 'electronhub':
            loadElectronHubModels();
            break;
        case 'openrouter':
            loadOpenRouterModels();
            break;
        case 'chutes':
            loadChutesModels();
            break;
        case 'nanogpt':
            loadNanoGPTModels();
            break;
        case 'siliconflow':
            loadSiliconFlowModels();
            break;
    }
}

async function loadChutesModels() {
    try {
        const response = await fetch('/api/openai/chutes/models/embedding', {
            method: 'POST',
            headers: getRequestHeaders({ omitContentType: true }),
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        /** @type {Array<any>} */
        const data = await response.json();
        const models = Array.isArray(data) ? data : [];
        populateChutesModelSelect(models);
    } catch (err) {
        console.warn('Chutes models fetch failed', err);
        populateChutesModelSelect([]);
    }
}

function populateChutesModelSelect(models) {
    const select = $('#vectors_chutes_model');
    select.empty();
    for (const m of models) {
        const option = document.createElement('option');
        option.value = m.slug;
        option.text = m.name;
        select.append(option);
    }
    if (!settings.chutes_model && models.length) {
        settings.chutes_model = models[0].slug;
    }
    $('#vectors_chutes_model').val(settings.chutes_model);
}

async function loadNanoGPTModels() {
    try {
        const response = await fetch('/api/openai/nanogpt/models/embedding', {
            method: 'POST',
            headers: getRequestHeaders({ omitContentType: true }),
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        /** @type {Array<any>} */
        const data = await response.json();
        const models = Array.isArray(data) ? data : [];
        populateNanoGPTModelSelect(models);
    } catch (err) {
        console.warn('NanoGPT models fetch failed', err);
        populateNanoGPTModelSelect([]);
    }
}

function populateNanoGPTModelSelect(models) {
    const select = $('#vectors_nanogpt_model');
    select.empty();
    for (const m of models) {
        const option = document.createElement('option');
        option.value = m.id;
        option.text = m.name || m.id;
        select.append(option);
    }
    if (!settings.nanogpt_model && models.length) {
        settings.nanogpt_model = models[0].id;
    }
    $('#vectors_nanogpt_model').val(settings.nanogpt_model);
}

async function loadElectronHubModels() {
    try {
        const response = await fetch('/api/openai/electronhub/models', {
            method: 'POST',
            headers: getRequestHeaders({ omitContentType: true }),
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        /** @type {Array<any>} */
        const data = await response.json();
        // filter by embeddings endpoint
        const models = Array.isArray(data) ? data.filter(m => Array.isArray(m?.endpoints) && m.endpoints.includes('/v1/embeddings')) : [];
        populateElectronHubModelSelect(models);
    } catch (err) {
        console.warn('Electron Hub models fetch failed', err);
        populateElectronHubModelSelect([]);
    }
}

/**
 * Populates the Electron Hub model select element.
 * @param {{ id: string, name: string }[]} models Electron Hub models
 */
function populateElectronHubModelSelect(models) {
    const select = $('#vectors_electronhub_model');
    select.empty();
    for (const m of models) {
        const option = document.createElement('option');
        option.value = m.id;
        option.text = m.name || m.id;
        select.append(option);
    }
    if (!settings.electronhub_model && models.length) {
        settings.electronhub_model = models[0].id;
    }
    $('#vectors_electronhub_model').val(settings.electronhub_model);
}

async function loadOpenRouterModels() {
    try {
        const response = await fetch('/api/openrouter/models/embedding', {
            method: 'POST',
            headers: getRequestHeaders({ omitContentType: true }),
        });
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }
        /** @type {Array<any>} */
        const data = await response.json();
        const models = Array.isArray(data) ? data : [];
        populateOpenRouterModelSelect(models);
    } catch (err) {
        console.warn('OpenRouter models fetch failed', err);
        populateOpenRouterModelSelect([]);
    }
}

/**
 * Populates the OpenRouter model select element.
 * @param {{ id: string, name: string }[]} models OpenRouter models
 */
function populateOpenRouterModelSelect(models) {
    const select = $('#vectors_openrouter_model');
    select.empty();
    for (const m of models) {
        const option = document.createElement('option');
        option.value = m.id;
        option.text = m.name || m.id;
        select.append(option);
    }
    if (!settings.openrouter_model && models.length) {
        settings.openrouter_model = models[0].id;
    }
    $('#vectors_openrouter_model').val(settings.openrouter_model);
}

async function loadSiliconFlowModels() {
    try {
        const response = await fetch('/api/openai/siliconflow/models/embedding', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                siliconflow_endpoint: oai_settings.siliconflow_endpoint,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}`);
        }

        /** @type {Array<any>} */
        const data = await response.json();
        const models = Array.isArray(data) ? data : [];
        populateSiliconFlowModelSelect(models);
    } catch (err) {
        console.warn('SiliconFlow models fetch failed', err);
        populateSiliconFlowModelSelect([]);
    }
}

function populateSiliconFlowModelSelect(models) {
    const select = $('#vectors_siliconflow_model');
    select.empty();
    for (const m of models) {
        const option = document.createElement('option');
        option.value = m.id;
        option.text = m.id;
        select.append(option);
    }
    if (!settings.siliconflow_model && models.length) {
        settings.siliconflow_model = models[0].id;
    }
    $('#vectors_siliconflow_model').val(settings.siliconflow_model);
}

/**
 * Executes a function with WebLLM error handling.
 * @param {function(): Promise<T>} func Function to execute
 * @returns {Promise<T>}
 * @template T
 */
async function executeWithWebLlmErrorHandling(func) {
    try {
        return await func();
    } catch (error) {
        console.log('Vectors: Failed to load WebLLM models', error);
        if (!(error instanceof Error)) {
            return;
        }
        switch (error.cause) {
            case 'webllm-not-available':
                toastr.warning('WebLLM is not available. Please install the extension.', 'WebLLM not installed');
                break;
            case 'webllm-not-updated':
                toastr.warning('The installed extension version does not support embeddings.', 'WebLLM update required');
                break;
        }
    }
}

/**
 * Loads and displays WebLLM models in the settings.
 * @returns {Promise<void>}
 */
function loadWebLlmModels() {
    return executeWithWebLlmErrorHandling(() => {
        const models = webllmProvider.getModels();
        $('#vectors_webllm_model').empty();
        for (const model of models) {
            $('#vectors_webllm_model').append($('<option>', { value: model.id, text: model.toString() }));
        }
        if (!settings.webllm_model || !models.some(x => x.id === settings.webllm_model)) {
            if (models.length) {
                settings.webllm_model = models[0].id;
            }
        }
        $('#vectors_webllm_model').val(settings.webllm_model);
        return Promise.resolve();
    });
}

/**
 * Creates WebLLM embeddings for a list of items.
 * @param {string[]} items Items to embed
 * @returns {Promise<Record<string, number[]>>} Calculated embeddings
 */
async function createWebLlmEmbeddings(items) {
    if (items.length === 0) {
        return /** @type {Record<string, number[]>} */ ({});
    }
    return executeWithWebLlmErrorHandling(async () => {
        const embeddings = await webllmProvider.embedTexts(items, settings.webllm_model);
        const result = /** @type {Record<string, number[]>} */ ({});
        for (let i = 0; i < items.length; i++) {
            result[items[i]] = embeddings[i];
        }
        return result;
    });
}

/**
 * Creates KoboldCpp embeddings for a list of items.
 * @param {string[]} items Items to embed
 * @returns {Promise<{embeddings: Record<string, number[]>, model: string}>} Calculated embeddings
 */
async function createKoboldCppEmbeddings(items) {
    const response = await fetch('/api/backends/kobold/embed', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            items: items,
            server: settings.use_alt_endpoint ? settings.alt_endpoint_url : textgenerationwebui_settings.server_urls[textgen_types.KOBOLDCPP],
        }),
    });

    if (!response.ok) {
        throw new Error('Failed to get KoboldCpp embeddings');
    }

    const data = await response.json();
    if (!Array.isArray(data.embeddings) || !data.model || data.embeddings.length !== items.length) {
        throw new Error('Invalid response from KoboldCpp embeddings');
    }

    const embeddings = /** @type {Record<string, number[]>} */ ({});
    for (let i = 0; i < data.embeddings.length; i++) {
        if (!Array.isArray(data.embeddings[i]) || data.embeddings[i].length === 0) {
            throw new Error('KoboldCpp returned an empty embedding. Reduce the chunk size and/or size threshold and try again.');
        }

        embeddings[items[i]] = data.embeddings[i];
    }

    return {
        embeddings: embeddings,
        model: data.model,
    };
}

async function onPurgeClick() {
    const chatId = getCurrentChatId();
    if (!chatId) {
        toastr.info('No chat selected', 'Purge aborted');
        return;
    }
    if (await purgeVectorIndex(chatId)) {
        toastr.success('Vector index purged', 'Purge successful');
    } else {
        toastr.error('Failed to purge vector index', 'Purge failed');
    }
}

async function onViewStatsClick() {
    const chatId = getCurrentChatId();
    if (!chatId) {
        toastr.info('No chat selected');
        return;
    }

    const hashesInCollection = await getSavedHashes(chatId);
    const totalHashes = hashesInCollection.length;
    const uniqueHashes = hashesInCollection.filter(onlyUnique).length;

    toastr.info(`Total hashes: <b>${totalHashes}</b><br>
    Unique hashes: <b>${uniqueHashes}</b><br><br>
    I'll mark collected messages with a green circle.`,
    `Stats for chat ${escapeHtml(chatId)}`,
    { timeOut: 10000, escapeHtml: false },
    );

    const chat = getContext().chat;
    for (const message of chat) {
        if (hashesInCollection.includes(getStringHash(substituteParams(message.mes)))) {
            const messageElement = $(`.mes[mesid="${chat.indexOf(message)}"]`);
            messageElement.addClass('vectorized');
        }
    }
}

async function onVectorizeAllFilesClick() {
    try {
        const dataBank = getDataBankAttachments();
        const chatAttachments = getContext().chat.filter(x => Array.isArray(x.extra?.files)).map(x => x.extra.files).flat();
        const allFiles = [...dataBank, ...chatAttachments];

        /**
         * Gets the chunk size for a file attachment.
         * @param file {import('../../chats.js').FileAttachment} File attachment
         * @returns {number} Chunk size for the file
         */
        function getChunkSize(file) {
            if (chatAttachments.includes(file)) {
                // Convert kilobytes to string length
                const thresholdLength = settings.size_threshold * 1024;
                return file.size > thresholdLength ? settings.chunk_size : -1;
            }

            if (dataBank.includes(file)) {
                // Convert kilobytes to string length
                const thresholdLength = settings.size_threshold_db * 1024;
                // Use chunk size from settings if file is larger than threshold
                return file.size > thresholdLength ? settings.chunk_size_db : -1;
            }

            return -1;
        }

        /**
         * Gets the overlap percent for a file attachment.
         * @param file {import('../../chats.js').FileAttachment} File attachment
         * @returns {number} Overlap percent for the file
         */
        function getOverlapPercent(file) {
            if (chatAttachments.includes(file)) {
                return settings.overlap_percent;
            }

            if (dataBank.includes(file)) {
                return settings.overlap_percent_db;
            }

            return 0;
        }

        let allSuccess = true;

        for (const file of allFiles) {
            const text = await getFileAttachment(file.url);
            const collectionId = getFileCollectionId(file.url);
            const hashes = await getSavedHashes(collectionId);

            if (hashes.length) {
                console.log(`Vectors: File ${file.name} is already vectorized`);
                continue;
            }

            const chunkSize = getChunkSize(file);
            const overlapPercent = getOverlapPercent(file);
            const result = await vectorizeFile(text, file.name, collectionId, chunkSize, overlapPercent);

            if (!result) {
                allSuccess = false;
            }
        }

        if (allSuccess) {
            toastr.success('All files vectorized', 'Vectorization successful');
        } else {
            toastr.warning('Some files failed to vectorize. Check browser console for more details.', 'Vector Storage');
        }
    } catch (error) {
        console.error('Vectors: Failed to vectorize all files', error);
        toastr.error('Failed to vectorize all files', 'Vectorization failed');
    }
}

async function onPurgeFilesClick() {
    try {
        const dataBank = getDataBankAttachments();
        const chatAttachments = getContext().chat.filter(x => Array.isArray(x.extra?.files)).map(x => x.extra.files).flat();
        const allFiles = [...dataBank, ...chatAttachments];

        for (const file of allFiles) {
            await purgeFileVectorIndex(file.url);
        }

        toastr.success('All files purged', 'Purge successful');
    } catch (error) {
        console.error('Vectors: Failed to purge all files', error);
        toastr.error('Failed to purge all files', 'Purge failed');
    }
}

async function activateWorldInfo(chat) {
    if (!settings.enabled_world_info) {
        console.debug('Vectors: Disabled for World Info');
        return;
    }

    const entries = await getSortedEntries();

    if (!Array.isArray(entries) || entries.length === 0) {
        console.debug('Vectors: No WI entries found');
        return;
    }

    // Group entries by "world" field
    const groupedEntries = {};

    for (const entry of entries) {
        // Skip orphaned entries. Is it even possible?
        if (!entry.world) {
            console.debug('Vectors: Skipped orphaned WI entry', entry);
            continue;
        }

        // Skip disabled entries
        if (entry.disable) {
            console.debug('Vectors: Skipped disabled WI entry', entry);
            continue;
        }

        // Skip entries without content
        if (!entry.content) {
            console.debug('Vectors: Skipped WI entry without content', entry);
            continue;
        }

        // Skip non-vectorized entries
        if (!entry.vectorized && !settings.enabled_for_all) {
            console.debug('Vectors: Skipped non-vectorized WI entry', entry);
            continue;
        }

        if (!Object.hasOwn(groupedEntries, entry.world)) {
            groupedEntries[entry.world] = [];
        }

        groupedEntries[entry.world].push(entry);
    }

    const collectionIds = [];

    if (Object.keys(groupedEntries).length === 0) {
        console.debug('Vectors: No WI entries to synchronize');
        return;
    }

    // Synchronize collections
    for (const world in groupedEntries) {
        const collectionId = `world_${getStringHash(world)}`;
        const hashesInCollection = await getSavedHashes(collectionId);
        const newEntries = groupedEntries[world].filter(x => !hashesInCollection.includes(getStringHash(x.content)));
        const deletedHashes = hashesInCollection.filter(x => !groupedEntries[world].some(y => getStringHash(y.content) === x));

        if (newEntries.length > 0) {
            console.log(`Vectors: Found ${newEntries.length} new WI entries for world ${world}`);
            await insertVectorItems(collectionId, newEntries.map(x => ({ hash: getStringHash(x.content), text: x.content, index: x.uid })));
        }

        if (deletedHashes.length > 0) {
            console.log(`Vectors: Deleted ${deletedHashes.length} old hashes for world ${world}`);
            await deleteVectorItems(collectionId, deletedHashes);
        }

        collectionIds.push(collectionId);
    }

    // Perform a multi-query
    const queryText = await getQueryText(chat, 'world-info');

    if (queryText.length === 0) {
        console.debug('Vectors: No text to query for WI');
        return;
    }

    const { topK: recallTopK, threshold: recallThreshold } = getRecallParams(settings.max_entries);
    const startedAt = Date.now();
    const queryResults = await queryMultipleCollections(collectionIds, queryText, recallTopK, recallThreshold);

    /** @type {number[]} */
    let activatedHashes;
    /** @type {{ rank: number, rerankScore: number | null, inserted: boolean, reason: string, text: string }[]} */
    let reportCandidates;

    if (shouldRerank()) {
        // Pool candidate texts across collections, keyed by hash to avoid duplicates
        const pooledHashes = [];
        const pooledTexts = [];
        for (const collectionId in queryResults) {
            for (const item of (queryResults[collectionId].metadata || [])) {
                const hash = Number(item.hash);
                if (!pooledHashes.includes(hash)) {
                    pooledHashes.push(hash);
                    pooledTexts.push(String(item.text ?? ''));
                }
            }
        }

        const rerankedOrder = await rerankTexts(queryText, pooledTexts);
        reportCandidates = rerankedOrder.map((entry, rank) => ({
            rank: rank,
            rerankScore: entry.score,
            inserted: rank < settings.max_entries,
            reason: rank < settings.max_entries ? '' : 'final count cut',
            text: String(pooledTexts[entry.index] ?? '').slice(0, RAG_LOG_PREVIEW_LENGTH),
        }));
        activatedHashes = rerankedOrder.slice(0, settings.max_entries).map(entry => pooledHashes[entry.index]);
    } else {
        activatedHashes = Object.values(queryResults).flatMap(x => x.hashes).filter(onlyUnique);
        reportCandidates = [];
    }

    recordRecallReport({
        ts: Date.now(),
        path: 'world_info',
        queryPreview: queryText,
        topK: recallTopK,
        threshold: recallThreshold ?? settings.score_threshold,
        reranked: shouldRerank(),
        tookMs: Date.now() - startedAt,
        finalCount: Math.min(activatedHashes.length, shouldRerank() ? settings.max_entries : activatedHashes.length),
        filteredByThreshold: [],
        candidates: reportCandidates,
    });

    const activatedEntries = [];

    // Activate entries found in the query results
    for (const entry of entries) {
        const hash = getStringHash(entry.content);

        if (activatedHashes.includes(hash)) {
            activatedEntries.push(entry);
        }
    }

    if (activatedEntries.length === 0) {
        console.debug('Vectors: No activated WI entries found');
        return;
    }

    console.log(`Vectors: Activated ${activatedEntries.length} WI entries`, activatedEntries);
    await eventSource.emit(event_types.WORLDINFO_FORCE_ACTIVATE, activatedEntries);
}

jQuery(async () => {
    if (!extension_settings.vectors) {
        extension_settings.vectors = settings;
    }

    // Migrate from old settings
    if (settings.enabled) {
        settings.enabled_chats = true;
    }

    Object.assign(settings, extension_settings.vectors);

    // Migrate from TensorFlow to Transformers
    settings.source = settings.source !== 'local' ? settings.source : 'transformers';

    // Restore diagnostics state (log ring buffer + last recall report)
    if (Array.isArray(extension_settings.vectors.debug_log)) {
        ragLogEntries.push(...extension_settings.vectors.debug_log.slice(-RAG_LOG_MAX_ENTRIES));
    }
    lastRecallReport = extension_settings.vectors.debug_last_report || null;

    // Load the settings template relative to this module's own URL.
    // The bundled renderExtensionTemplateAsync assumes a non-third-party folder layout.
    const template = await renderTemplateAsync(new URL('settings.html', import.meta.url).href, {}, true, true, true);
    $('#vectors_container').append(template);
    $('#vectors_enabled_chats').prop('checked', settings.enabled_chats).on('input', () => {
        settings.enabled_chats = $('#vectors_enabled_chats').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
        toggleSettings();
    });
    $('#vectors_enabled_files').prop('checked', settings.enabled_files).on('input', () => {
        settings.enabled_files = $('#vectors_enabled_files').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
        toggleSettings();
    });
    $('#vectors_source').val(settings.source).on('change', () => {
        settings.source = String($('#vectors_source').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
        toggleSettings();
    });
    $('#vector_altEndpointUrl_enabled').prop('checked', settings.use_alt_endpoint).on('input', () => {
        settings.use_alt_endpoint = $('#vector_altEndpointUrl_enabled').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vector_altEndpoint_address').val(settings.alt_endpoint_url).on('change', () => {
        settings.alt_endpoint_url = String($('#vector_altEndpoint_address').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_togetherai_model').val(settings.togetherai_model).on('change', () => {
        settings.togetherai_model = String($('#vectors_togetherai_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_openai_model').val(settings.openai_model).on('change', () => {
        settings.openai_model = String($('#vectors_openai_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_electronhub_model').val(settings.electronhub_model).on('change', () => {
        settings.electronhub_model = String($('#vectors_electronhub_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_chutes_model').val(settings.chutes_model).on('change', () => {
        settings.chutes_model = String($('#vectors_chutes_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_nanogpt_model').val(settings.nanogpt_model).on('change', () => {
        settings.nanogpt_model = String($('#vectors_nanogpt_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_siliconflow_model').val(settings.siliconflow_model).on('change', () => {
        settings.siliconflow_model = String($('#vectors_siliconflow_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_openrouter_model').val(settings.openrouter_model).on('change', () => {
        settings.openrouter_model = String($('#vectors_openrouter_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_cohere_model').val(settings.cohere_model).on('change', () => {
        settings.cohere_model = String($('#vectors_cohere_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_ollama_model').val(settings.ollama_model).on('input', () => {
        settings.ollama_model = String($('#vectors_ollama_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_vllm_model').val(settings.vllm_model).on('input', () => {
        settings.vllm_model = String($('#vectors_vllm_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_ollama_keep').prop('checked', settings.ollama_keep).on('input', () => {
        settings.ollama_keep = $('#vectors_ollama_keep').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_template').val(settings.template).on('input', () => {
        settings.template = String($('#vectors_template').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_depth').val(settings.depth).on('input', () => {
        settings.depth = Number($('#vectors_depth').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_protect').val(settings.protect).on('input', () => {
        settings.protect = Number($('#vectors_protect').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_insert').val(settings.insert).on('input', () => {
        settings.insert = Number($('#vectors_insert').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_query').val(settings.query).on('input', () => {
        settings.query = Number($('#vectors_query').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $(`input[name="vectors_position"][value="${settings.position}"]`).prop('checked', true);
    $('input[name="vectors_position"]').on('change', () => {
        settings.position = Number($('input[name="vectors_position"]:checked').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_vectorize_all').on('click', onVectorizeAllClick);
    $('#vectors_purge').on('click', onPurgeClick);
    $('#vectors_view_stats').on('click', onViewStatsClick);
    $('#vectors_files_vectorize_all').on('click', onVectorizeAllFilesClick);
    $('#vectors_files_purge').on('click', onPurgeFilesClick);

    $('#vectors_size_threshold').val(settings.size_threshold).on('input', () => {
        settings.size_threshold = Number($('#vectors_size_threshold').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_chunk_size').val(settings.chunk_size).on('input', () => {
        settings.chunk_size = Number($('#vectors_chunk_size').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_chunk_count').val(settings.chunk_count).on('input', () => {
        settings.chunk_count = Number($('#vectors_chunk_count').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_include_wi').prop('checked', settings.include_wi).on('input', () => {
        settings.include_wi = !!$('#vectors_include_wi').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_summarize').prop('checked', settings.summarize).on('input', () => {
        settings.summarize = !!$('#vectors_summarize').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_summarize_user').prop('checked', settings.summarize_sent).on('input', () => {
        settings.summarize_sent = !!$('#vectors_summarize_user').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_summary_source').val(settings.summary_source).on('change', () => {
        settings.summary_source = String($('#vectors_summary_source').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_summary_prompt').val(settings.summary_prompt).on('input', () => {
        settings.summary_prompt = String($('#vectors_summary_prompt').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_message_chunk_size').val(settings.message_chunk_size).on('input', () => {
        settings.message_chunk_size = Number($('#vectors_message_chunk_size').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_size_threshold_db').val(settings.size_threshold_db).on('input', () => {
        settings.size_threshold_db = Number($('#vectors_size_threshold_db').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_chunk_size_db').val(settings.chunk_size_db).on('input', () => {
        settings.chunk_size_db = Number($('#vectors_chunk_size_db').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_chunk_count_db').val(settings.chunk_count_db).on('input', () => {
        settings.chunk_count_db = Number($('#vectors_chunk_count_db').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_overlap_percent').val(settings.overlap_percent).on('input', () => {
        settings.overlap_percent = Number($('#vectors_overlap_percent').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_overlap_percent_db').val(settings.overlap_percent_db).on('input', () => {
        settings.overlap_percent_db = Number($('#vectors_overlap_percent_db').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_file_template_db').val(settings.file_template_db).on('input', () => {
        settings.file_template_db = String($('#vectors_file_template_db').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $(`input[name="vectors_file_position_db"][value="${settings.file_position_db}"]`).prop('checked', true);
    $('input[name="vectors_file_position_db"]').on('change', () => {
        settings.file_position_db = Number($('input[name="vectors_file_position_db"]:checked').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_file_depth_db').val(settings.file_depth_db).on('input', () => {
        settings.file_depth_db = Number($('#vectors_file_depth_db').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_file_depth_role_db').val(settings.file_depth_role_db).on('input', () => {
        settings.file_depth_role_db = Number($('#vectors_file_depth_role_db').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_translate_files').prop('checked', settings.translate_files).on('input', () => {
        settings.translate_files = !!$('#vectors_translate_files').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enabled_world_info').prop('checked', settings.enabled_world_info).on('input', () => {
        settings.enabled_world_info = !!$('#vectors_enabled_world_info').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
        toggleSettings();
    });

    $('#vectors_enabled_for_all').prop('checked', settings.enabled_for_all).on('input', () => {
        settings.enabled_for_all = !!$('#vectors_enabled_for_all').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_max_entries').val(settings.max_entries).on('input', () => {
        settings.max_entries = Number($('#vectors_max_entries').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_score_threshold').val(settings.score_threshold).on('input', () => {
        settings.score_threshold = Number($('#vectors_score_threshold').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    // ============================= Reranker settings =============================

    $('#vectors_rerank_enabled').prop('checked', settings.rerank_enabled).on('input', () => {
        settings.rerank_enabled = !!$('#vectors_rerank_enabled').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
        toggleSettings();
    });
    $('#vectors_rerank_url').val(settings.rerank_url).on('change', () => {
        settings.rerank_url = String($('#vectors_rerank_url').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_rerank_key').val(settings.rerank_key).on('change', () => {
        settings.rerank_key = String($('#vectors_rerank_key').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_rerank_model').val(settings.rerank_model).on('change', () => {
        settings.rerank_model = String($('#vectors_rerank_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_rerank_recall').val(settings.rerank_recall).on('input', () => {
        settings.rerank_recall = Number($('#vectors_rerank_recall').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_rerank_ignore_threshold').prop('checked', settings.rerank_ignore_threshold).on('input', () => {
        settings.rerank_ignore_threshold = !!$('#vectors_rerank_ignore_threshold').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_rerank_query_max_chars').val(settings.rerank_query_max_chars).on('input', () => {
        settings.rerank_query_max_chars = Number($('#vectors_rerank_query_max_chars').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_rerank_doc_max_chars').val(settings.rerank_doc_max_chars).on('input', () => {
        settings.rerank_doc_max_chars = Number($('#vectors_rerank_doc_max_chars').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_rerank_timeout_ms').val(settings.rerank_timeout_ms).on('input', () => {
        settings.rerank_timeout_ms = Number($('#vectors_rerank_timeout_ms').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_rerank_test').on('click', async () => {
        const button = $('#vectors_rerank_test');
        button.addClass('busy');
        try {
            const result = await testRerankerConnection();
            if (result.ok) {
                toastr.success(result.message, 'Reranker Test');
                ragLogEvent('rerank', 'info', `Connection test OK`, { ms: result.ms });
            } else {
                toastr.error(result.message, 'Reranker Test');
                ragLogEvent('rerank', 'error', `Connection test failed: ${result.message}`);
            }
        } finally {
            button.removeClass('busy');
        }
    });

    // ============================== Diagnostics =================================

    $('#vectors_debug_diff_threshold').prop('checked', settings.debug_diff_threshold).on('input', () => {
        settings.debug_diff_threshold = !!$('#vectors_debug_diff_threshold').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });
    $('#vectors_diag_filter').on('change', renderRagLog);
    $('#vectors_view_report').on('click', () => {
        renderLastReport();
        $('#vectors_last_report').toggle();
    });
    $('#vectors_export_logs').on('click', () => {
        const lines = ragLogEntries.map(x => JSON.stringify(x)).join('\n');
        const blob = new Blob([lines || '[]'], { type: 'application/x-ndjson' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `vectors-debug-${new Date().toISOString().replace(/[:.]/g, '-')}.jsonl`;
        anchor.click();
        URL.revokeObjectURL(url);
    });
    $('#vectors_clear_logs').on('click', () => {
        ragLogEntries.length = 0;
        lastRecallReport = null;
        extension_settings.vectors.debug_log = [];
        extension_settings.vectors.debug_last_report = null;
        saveSettingsDebounced();
        renderRagLog();
        renderLastReport();
        updateStatusBadge('(idle)', 'info');
    });
    $('#vectors_run_diagnostics').on('click', async () => {
        const button = $('#vectors_run_diagnostics');
        button.addClass('busy');
        updateStatusBadge('…running diagnostics', 'warn');

        /** @type {{ name: string, ok: boolean, info: string }[]} */
        const results = [];
        try {
            // Probe 1: embedding + vector query roundtrip via the ST server
            try {
                const chatId = getCurrentChatId();
                if (!chatId) {
                    throw new Error('no chat selected');
                }
                const startedAt = Date.now();
                await queryCollection(chatId, 'vector storage diagnostic probe', 1, 0);
                results.push({ name: 'Embedding+Query', ok: true, info: `roundtrip ${Date.now() - startedAt}ms` });
            } catch (error) {
                results.push({ name: 'Embedding+Query', ok: false, info: String(error?.message || error) });
            }

            // Probe 2: reranker connectivity
            if (settings.rerank_enabled) {
                const rerankResult = await testRerankerConnection();
                results.push({ name: 'Reranker', ok: rerankResult.ok, info: rerankResult.message });
            } else {
                results.push({ name: 'Reranker', ok: true, info: 'disabled' });
            }

            // Probe 3: chunk splitter integrity with a multibyte sample
            try {
                const sample = '第一段😀emoji测试🎉包含中文and English mixed text。第二段落继续直到被切开';
                const pieces = splitRecursive(sample, 7, []);
                const broken = pieces.filter(piece => LONE_SURROGATE_REGEX.test(String(piece)));
                results.push({
                    name: 'Chunk splitter',
                    ok: broken.length === 0,
                    info: broken.length === 0
                        ? `${pieces.length} pieces, no broken surrogates`
                        : `${broken.length}/${pieces.length} pieces contain BROKEN emoji (lone surrogates)! Keep emoji cleaning upstream.`,
                });
            } catch (error) {
                results.push({ name: 'Chunk splitter', ok: false, info: String(error?.message || error) });
            }

            const failedCount = results.filter(x => !x.ok).length;
            const summary = results.map(x => `${x.ok ? '✓' : '✗'} <b>${escapeHtml(x.name)}</b>: ${escapeHtml(x.info)}`).join('<br>');
            const level = failedCount > 0 ? 'warning' : 'success';
            toastr[level](summary, 'Vector Storage Diagnostics', { timeOut: 15000, escapeHtml: false });
            for (const result of results) {
                ragLogEvent('system', result.ok ? 'info' : 'warn', `Diagnostics [${result.name}]: ${result.ok ? 'OK' : 'FAILED'}`, { info: result.info });
            }
            updateStatusBadge(failedCount > 0 ? `⚠ ${failedCount} check(s) failed` : '✓ all checks passed', failedCount > 0 ? 'warn' : 'info');
        } finally {
            button.removeClass('busy');
        }
    });

    $('#vectors_force_chunk_delimiter').val(settings.force_chunk_delimiter).on('input', () => {
        settings.force_chunk_delimiter = String($('#vectors_force_chunk_delimiter').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_only_custom_boundary').prop('checked', settings.only_custom_boundary).on('input', () => {
        settings.only_custom_boundary = !!$('#vectors_only_custom_boundary').prop('checked');
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_ollama_pull').on('click', (e) => {
        const presetModel = extension_settings.vectors.ollama_model || '';
        e.preventDefault();
        $('#ollama_download_model').trigger('click');
        $('#dialogue_popup_input').val(presetModel);
    });

    $('#vectors_webllm_install').on('click', (e) => {
        e.preventDefault();
        e.stopPropagation();

        if (Object.hasOwn(SillyTavern, 'llm')) {
            toastr.info('WebLLM is already installed');
            return;
        }

        openThirdPartyExtensionMenu('https://github.com/SillyTavern/Extension-WebLLM');
    });

    $('#vectors_webllm_model').on('input', () => {
        settings.webllm_model = String($('#vectors_webllm_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#vectors_webllm_load').on('click', async () => {
        if (!settings.webllm_model) return;
        await webllmProvider.loadModel(settings.webllm_model);
        toastr.success('WebLLM model loaded');
    });

    $('#vectors_google_model').val(settings.google_model).on('input', () => {
        settings.google_model = String($('#vectors_google_model').val());
        Object.assign(extension_settings.vectors, settings);
        saveSettingsDebounced();
    });

    $('#api_key_nomicai').toggleClass('success', !!secret_state[SECRET_KEYS.NOMICAI]);
    [event_types.SECRET_WRITTEN, event_types.SECRET_DELETED, event_types.SECRET_ROTATED].forEach(event => {
        eventSource.on(event, (/** @type {string} */ key) => {
            if (key !== SECRET_KEYS.NOMICAI) return;
            $('#api_key_nomicai').toggleClass('success', !!secret_state[SECRET_KEYS.NOMICAI]);
        });
    });

    toggleSettings();
    renderRagLog();
    renderLastReport();
    updateStatusBadge(ragLogEntries.length ? `(last session: ${ragLogEntries.length} log entries)` : '(idle)', 'info');
    eventSource.on(event_types.MESSAGE_DELETED, onChatEvent);
    eventSource.on(event_types.MESSAGE_EDITED, onChatEvent);
    eventSource.on(event_types.MESSAGE_SENT, onChatEvent);
    eventSource.on(event_types.MESSAGE_RECEIVED, onChatEvent);
    eventSource.on(event_types.MESSAGE_SWIPED, onChatEvent);
    eventSource.on(event_types.CHAT_DELETED, purgeVectorIndex);
    eventSource.on(event_types.GROUP_CHAT_DELETED, purgeVectorIndex);
    eventSource.on(event_types.FILE_ATTACHMENT_DELETED, purgeFileVectorIndex);
    eventSource.on(event_types.EXTENSION_SETTINGS_LOADED, async (manifest) => {
        if (settings.source === 'webllm' && manifest?.display_name === 'WebLLM') {
            await loadWebLlmModels();
        }
    });

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'db-ingest',
        callback: async () => {
            await ingestDataBankAttachments();
            return '';
        },
        aliases: ['databank-ingest', 'data-bank-ingest'],
        helpString: 'Force the ingestion of all Data Bank attachments.',
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'db-purge',
        callback: async () => {
            const dataBank = getDataBankAttachments();

            for (const file of dataBank) {
                await purgeFileVectorIndex(file.url);
            }

            return '';
        },
        aliases: ['databank-purge', 'data-bank-purge'],
        helpString: 'Purge the vector index for all Data Bank attachments.',
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'db-search',
        callback: async (args, query) => {
            const clamp = (v) => Number.isNaN(v) ? null : Math.min(1, Math.max(0, v));
            const threshold = clamp(Number(args?.threshold ?? settings.score_threshold));
            const validateCount = (v) => Number.isNaN(v) || !Number.isInteger(v) || v < 1 ? null : v;
            const count = validateCount(Number(args?.count)) ?? settings.chunk_count_db;
            const source = String(args?.source ?? '');
            const attachments = source ? getDataBankAttachmentsForSource(source, false) : getDataBankAttachments(false);
            const collectionIds = await ingestDataBankAttachments(String(source));
            const queryResults = await queryMultipleCollections(collectionIds, String(query), count, threshold);

            // Get URLs
            const urls = Object
                .keys(queryResults)
                .map(x => attachments.find(y => getFileCollectionId(y.url) === x))
                .filter(x => x)
                .map(x => x.url);

            // Gets the actual text content of chunks
            const getChunksText = () => {
                let textResult = '';
                for (const collectionId in queryResults) {
                    const metadata = queryResults[collectionId].metadata?.filter(x => x.text)?.sort((a, b) => a.index - b.index)?.map(x => x.text)?.filter(onlyUnique) || [];
                    textResult += metadata.join('\n') + '\n\n';
                }
                return textResult;
            };
            if (args.return === 'chunks') {
                return getChunksText();
            }

            // @ts-ignore
            return slashCommandReturnHelper.doReturn(args.return ?? 'object', urls, { objectToStringFunc: list => list.join('\n') });
        },
        aliases: ['databank-search', 'data-bank-search'],
        helpString: 'Search the Data Bank for a specific query using vector similarity. Returns a list of file URLs with the most relevant content.',
        namedArgumentList: [
            new SlashCommandNamedArgument('threshold', 'Threshold for the similarity score in the [0, 1] range. Uses the global config value if not set.', ARGUMENT_TYPE.NUMBER, false, false, ''),
            new SlashCommandNamedArgument('count', 'Maximum number of query results to return.', ARGUMENT_TYPE.NUMBER, false, false, ''),
            new SlashCommandNamedArgument('source', 'Optional filter for the attachments by source.', ARGUMENT_TYPE.STRING, false, false, '', ['global', 'character', 'chat']),
            SlashCommandNamedArgument.fromProps({
                name: 'return',
                description: 'How you want the return value to be provided',
                typeList: [ARGUMENT_TYPE.STRING],
                defaultValue: 'object',
                enumList: [
                    new SlashCommandEnumValue('chunks', 'Return the actual content chunks', enumTypes.enum, '{}'),
                    ...slashCommandReturnHelper.enumList({ allowObject: true }),
                ],
                forceEnum: true,
            }),
        ],
        unnamedArgumentList: [
            new SlashCommandArgument('Query to search by.', ARGUMENT_TYPE.STRING, true, false),
        ],
        returns: ARGUMENT_TYPE.LIST,
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'vector-threshold',
        helpString: 'Set the vector score threshold or return the current threshold if no argument is provided.',
        returns: 'score threshold value',
        unnamedArgumentList: [
            SlashCommandArgument.fromProps({
                description: 'Score threshold (number).',
                typeList: [ARGUMENT_TYPE.NUMBER],
            }),
        ],
        callback: async (_args, value) => {
            const raw = String(value ?? '').trim();
            if (!raw) {
                return String(settings.score_threshold);
            }

            const parsed = Number(raw);
            if (!Number.isFinite(parsed) || parsed < 0 || parsed > 1) {
                toastr.warning('Score threshold must be a number between 0 and 1.');
                return '';
            }

            $('#vectors_score_threshold')
                .val(parsed)
                .trigger('input');

            return String(settings.score_threshold);
        },
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'vector-query',
        helpString: 'Set the vector query messages or returns the current query messages count if no argument is provided',
        returns: 'the query messages value',
        unnamedArgumentList: [
            SlashCommandArgument.fromProps({
                description: 'Query messages (number > 0).',
                typeList: [ARGUMENT_TYPE.NUMBER],
            }),
        ],
        callback: async (_args, value) => {
            const raw = String(value ?? '').trim();
            if (!raw) {
                return String(settings.query);
            }

            const parsed = Number(raw);
            if (!Number.isFinite(parsed) || parsed <= 0) {
                toastr.warning('Query messages must be a number greater than 0.');
                return '';
            }

            $('#vectors_query')
                .val(parsed)
                .trigger('input');

            return String(settings.query);
        },
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'vector-max-entries',
        helpString: 'Set the vector world info max entries or returns the current max entries if no argument is provided',
        returns: 'world info max entries',
        unnamedArgumentList: [
            SlashCommandArgument.fromProps({
                description: 'Max entries (number > 0).',
                typeList: [ARGUMENT_TYPE.NUMBER],
            }),
        ],
        callback: async (_args, value) => {
            const raw = String(value ?? '').trim();
            if (!raw) {
                return String(settings.max_entries);
            }

            const parsed = Number(raw);
            if (!Number.isFinite(parsed) || parsed <= 0) {
                toastr.warning('Max entries must be a number greater than 0.');
                return '';
            }

            $('#vectors_max_entries')
                .val(parsed)
                .trigger('input');

            return String(settings.max_entries);
        },
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'vector-chats-state',
        helpString: 'Set whether chat vectorization is enabled or return the current boolean if no argument is provided',
        returns: 'boolean for if chat vectorization is enabled',
        unnamedArgumentList: [
            SlashCommandArgument.fromProps({
                description: 'boolean to set whether chat vectorization is enabled',
                typeList: [ARGUMENT_TYPE.BOOLEAN],
                enumList: commonEnumProviders.boolean('trueFalse')(),
            }),
        ],
        callback: async (_args, value) => {
            const raw = String(value ?? '').trim();
            if (!raw) {
                return String(settings.enabled_chats);
            }

            const parsed = isTrueBoolean(raw);
            $('#vectors_enabled_chats')
                .prop('checked', parsed)
                .trigger('input');

            return String(settings.enabled_chats);
        },
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'vector-files-state',
        helpString: 'Set whether file vectorization is enabled or return the current boolean if no argument is provided',
        returns: 'boolean for if file vectorization is enabled',
        unnamedArgumentList: [
            SlashCommandArgument.fromProps({
                description: 'boolean to set whether file vectorization is enabled',
                typeList: [ARGUMENT_TYPE.BOOLEAN],
                enumList: commonEnumProviders.boolean('trueFalse')(),
            }),
        ],
        callback: async (_args, value) => {
            const raw = String(value ?? '').trim();
            if (!raw) {
                return String(settings.enabled_files);
            }

            const parsed = isTrueBoolean(raw) ;
            $('#vectors_enabled_files')
                .prop('checked', parsed)
                .trigger('input');

            return String(settings.enabled_files);
        },
    }));

    SlashCommandParser.addCommandObject(SlashCommand.fromProps({
        name: 'vector-worldinfo-state',
        helpString: 'Set whether world info vectorization is enabled or return the current boolean if no argument is provided',
        returns: 'boolean for if world info vectorization is enabled',
        unnamedArgumentList: [
            SlashCommandArgument.fromProps({
                description: 'boolean to set whether world info vectorization is enabled',
                typeList: [ARGUMENT_TYPE.BOOLEAN],
                enumList: commonEnumProviders.boolean('trueFalse')(),
            }),
        ],
        callback: async (_args, value) => {
            const raw = String(value ?? '').trim();
            if (!raw) {
                return String(settings.enabled_world_info);
            }

            const parsed = isTrueBoolean(raw);
            $('#vectors_enabled_world_info')
                .prop('checked', parsed)
                .trigger('input');

            return String(settings.enabled_world_info);
        },
    }));

    registerDebugFunction('purge-everything', 'Purge all vector indices', 'Obliterate all stored vectors for all sources. No mercy.', async () => {
        if (!confirm('Are you sure?')) {
            return;
        }
        await purgeAllVectorIndexes();
    });
});
