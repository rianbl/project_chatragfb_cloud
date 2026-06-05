let apiUrl = '/chat';
const dataApiBaseUrl = '';

let contextState = {
    documents: [],
    limits: { max_documents: 3, max_total_size_bytes: 0 },
    usage: { document_count: 0, total_size_bytes: 0 },
    is_upload_blocked: false,
    blocked_reasons: []
};
let contextPanelCollapsed = false;
let contextUiStatus = 'idle';
let chatPanelCollapsed = false;
let memoryPanelCollapsed = false;
let memoryUiStatus = 'idle';
let memoryGraphEventSource = null;
let memoryRequestInFlight = false;
let memoryFocusMode = false;
const MAX_CONVERSATION_CONTEXT_MESSAGES = 12;
let conversationHistory = [];
let memoryGraphState = {
    nodes: [],
    edges: [],
    meta: { entity_count: 0, relation_count: 0, updated_at: '' }
};
let memoryNetwork = null;
const MEMORY_PANEL_WIDTH_STORAGE_KEY = 'memoryPanelWidthPx';
const MEMORY_PANEL_DEFAULT_WIDTH = 320;
const MEMORY_PANEL_MIN_WIDTH = 240;
const MEMORY_PANEL_MAX_WIDTH = 960;
const INTERNAL_MEMORY_NODE_IDS = new Set(['session_memory']);

function pushConversationEntry(role, content) {
    const normalizedRole = role === 'user' ? 'user' : 'assistant';
    const normalizedContent = String(content || '').trim();
    if (!normalizedContent) {
        return;
    }
    conversationHistory.push({ role: normalizedRole, content: normalizedContent });
    if (conversationHistory.length > MAX_CONVERSATION_CONTEXT_MESSAGES) {
        conversationHistory = conversationHistory.slice(-MAX_CONVERSATION_CONTEXT_MESSAGES);
    }
}

function buildConversationContextPayload() {
    return conversationHistory.map((item) => `${item.role}: ${item.content}`);
}

function setLoadingOverlayVisible(isVisible) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (!loadingOverlay) {
        return;
    }
    loadingOverlay.style.display = isVisible ? 'flex' : 'none';
}

function setContextPanelCollapsed(isCollapsed) {
    const filesPanel = document.getElementById('filesPanel');
    const toggleButton = document.getElementById('contextToggleButton');
    if (!filesPanel || !toggleButton) {
        return;
    }

    contextPanelCollapsed = Boolean(isCollapsed);
    filesPanel.classList.toggle('is-collapsed', contextPanelCollapsed);
    toggleButton.setAttribute('aria-expanded', String(!contextPanelCollapsed));
    toggleButton.textContent = contextPanelCollapsed ? '▶' : '◀';
    toggleButton.title = contextPanelCollapsed ? 'Expand context panel' : 'Collapse context panel';
}

function toggleContextPanel() {
    setContextPanelCollapsed(!contextPanelCollapsed);
}

function setChatPanelCollapsed(isCollapsed, options = {}) {
    const chatColumn = document.querySelector('.chat-column');
    if (!chatColumn) {
        return;
    }

    chatPanelCollapsed = Boolean(isCollapsed);
    chatColumn.classList.toggle('is-collapsed', chatPanelCollapsed);

    if (!chatPanelCollapsed && memoryFocusMode && !options.skipFocusReset) {
        setMemoryFocusMode(false, { skipChatSync: true });
    }
}

function setMemoryFocusMode(enabled, options = {}) {
    const workspaceRow = document.querySelector('.workspace-row');
    const memoryPanel = document.getElementById('memoryPanel');
    const focusButton = document.getElementById('memoryFocusButton');
    if (!workspaceRow || !memoryPanel || !focusButton) {
        return;
    }
    if (window.innerWidth <= 900 && enabled) {
        return;
    }
    if (memoryPanelCollapsed && enabled) {
        return;
    }

    memoryFocusMode = Boolean(enabled);
    workspaceRow.classList.toggle('memory-focus', memoryFocusMode);
    memoryPanel.classList.toggle('is-focus', memoryFocusMode);
    focusButton.setAttribute('aria-pressed', String(memoryFocusMode));
    focusButton.title = memoryFocusMode ? 'Return graph panel to default size' : 'Expand graph panel';
    focusButton.textContent = memoryFocusMode ? '⤡' : '⤢';

    if (!options.skipChatSync) {
        setChatPanelCollapsed(memoryFocusMode, { skipFocusReset: true });
    }

    if (!memoryFocusMode) {
        applyMemoryPanelWidth(MEMORY_PANEL_DEFAULT_WIDTH, true);
    }

    requestAnimationFrame(() => {
        if (memoryNetwork) {
            memoryNetwork.redraw();
            memoryNetwork.fit({ animation: { duration: 180, easingFunction: 'easeInOutQuad' } });
        }
    });
}

function setMemoryPanelCollapsed(isCollapsed) {
    const memoryPanel = document.getElementById('memoryPanel');
    const toggleButton = document.getElementById('memoryToggleButton');
    if (!memoryPanel || !toggleButton) {
        return;
    }

    if (Boolean(isCollapsed) && memoryFocusMode) {
        setMemoryFocusMode(false);
    }

    memoryPanelCollapsed = Boolean(isCollapsed);
    memoryPanel.classList.toggle('is-collapsed', memoryPanelCollapsed);
    toggleButton.setAttribute('aria-expanded', String(!memoryPanelCollapsed));
    toggleButton.textContent = memoryPanelCollapsed ? '◀' : '▶';
    toggleButton.title = memoryPanelCollapsed ? 'Expand knowledge graph panel' : 'Collapse knowledge graph panel';

    if (memoryPanelCollapsed) {
        stopMemoryGraphStream();
    } else {
        startMemoryGraphStream();
        if (!window.EventSource) {
            loadMemoryGraph({ silent: true }).catch((error) => console.error(error));
        }
        requestAnimationFrame(() => {
            if (memoryNetwork) {
                memoryNetwork.redraw();
                memoryNetwork.fit({ animation: { duration: 180, easingFunction: 'easeInOutQuad' } });
            }
        });
    }
}

function toggleMemoryPanel() {
    setMemoryPanelCollapsed(!memoryPanelCollapsed);
}

function toggleMemoryFocusMode() {
    setMemoryFocusMode(!memoryFocusMode);
}

function clampMemoryPanelWidth(width) {
    const numericWidth = Number(width);
    if (!Number.isFinite(numericWidth)) {
        return MEMORY_PANEL_DEFAULT_WIDTH;
    }
    return Math.max(MEMORY_PANEL_MIN_WIDTH, Math.min(MEMORY_PANEL_MAX_WIDTH, Math.round(numericWidth)));
}

function applyMemoryPanelWidth(width, persist = false) {
    const memoryPanel = document.getElementById('memoryPanel');
    if (!memoryPanel || window.innerWidth <= 900) {
        return;
    }
    const clampedWidth = clampMemoryPanelWidth(width);
    memoryPanel.style.setProperty('--memory-panel-width', `${clampedWidth}px`);
    if (persist) {
        try {
            localStorage.setItem(MEMORY_PANEL_WIDTH_STORAGE_KEY, String(clampedWidth));
        } catch {
            // Non-blocking: ignore storage errors.
        }
    }
    if (memoryNetwork) {
        memoryNetwork.redraw();
    }
}

function restoreMemoryPanelWidth() {
    let savedWidth = MEMORY_PANEL_DEFAULT_WIDTH;
    try {
        const fromStorage = localStorage.getItem(MEMORY_PANEL_WIDTH_STORAGE_KEY);
        if (fromStorage) {
            savedWidth = Number(fromStorage);
        }
    } catch {
        // Non-blocking: ignore storage errors.
    }
    applyMemoryPanelWidth(savedWidth, false);
}

function initializeMemoryPanelResize() {
    const memoryPanel = document.getElementById('memoryPanel');
    const resizeHandle = document.getElementById('memoryResizeHandle');
    if (!memoryPanel || !resizeHandle) {
        return;
    }

    let isDragging = false;
    let startX = 0;
    let startWidth = 0;

    const onPointerMove = (event) => {
        if (!isDragging) {
            return;
        }
        const deltaX = startX - event.clientX;
        const nextWidth = startWidth + deltaX;
        applyMemoryPanelWidth(nextWidth, false);
    };

    const stopDrag = () => {
        if (!isDragging) {
            return;
        }
        isDragging = false;
        memoryPanel.classList.remove('is-resizing');
        document.body.classList.remove('memory-panel-resizing');
        const appliedWidth = parseFloat(getComputedStyle(memoryPanel).width);
        applyMemoryPanelWidth(appliedWidth, true);
        window.removeEventListener('pointermove', onPointerMove);
        window.removeEventListener('pointerup', stopDrag);
        window.removeEventListener('pointercancel', stopDrag);
    };

    resizeHandle.addEventListener('pointerdown', (event) => {
        if (window.innerWidth <= 900 || memoryPanelCollapsed || memoryFocusMode) {
            return;
        }
        isDragging = true;
        startX = event.clientX;
        startWidth = memoryPanel.getBoundingClientRect().width;
        memoryPanel.classList.add('is-resizing');
        document.body.classList.add('memory-panel-resizing');
        window.addEventListener('pointermove', onPointerMove);
        window.addEventListener('pointerup', stopDrag);
        window.addEventListener('pointercancel', stopDrag);
        event.preventDefault();
    });
}

function getContextStatusIcon(documentCount) {
    if (contextUiStatus === 'error') {
        return '⚠️';
    }
    if (contextUiStatus === 'loading') {
        return '⏳';
    }
    if (Number(documentCount || 0) > 0) {
        return '📚';
    }
    return '📂';
}

function getMemoryStatusIcon(nodeCount) {
    if (memoryUiStatus === 'error') {
        return '⚠️';
    }
    if (memoryUiStatus === 'loading') {
        return '⏳';
    }
    if (Number(nodeCount || 0) > 0) {
        return '🧠';
    }
    return '🕸️';
}

function getFileTypeIcon(fileType) {
    const normalizedType = String(fileType || '').toLowerCase();
    if (normalizedType === 'pdf') {
        return '📄';
    }
    if (normalizedType === 'csv') {
        return '📊';
    }
    if (normalizedType === 'txt') {
        return '📝';
    }
    return '📁';
}

function getDocumentDetail(doc) {
    const normalizedType = String(doc.file_type || '').toLowerCase();

    if (normalizedType === 'pdf') {
        const pages = Number(doc.page_count || doc.pdf_pages_detected || 0);
        if (pages > 0) {
            return `${pages} ${pages === 1 ? 'page' : 'pages'}`;
        }
    }

    if (normalizedType === 'csv') {
        const rows = Number(doc.csv_rows || 0);
        if (rows > 0) {
            return `${rows} ${rows === 1 ? 'row' : 'rows'}`;
        }
    }

    if (normalizedType === 'txt') {
        const blocks = Number(doc.txt_blocks || 0);
        if (blocks > 0) {
            return `${blocks} ${blocks === 1 ? 'block' : 'blocks'}`;
        }
    }

    const chunks = Number(doc.chunk_count || 0);
    return `${chunks} ${chunks === 1 ? 'chunk' : 'chunks'}`;
}

async function getConfig() {
    const response = await fetch('config.json');
    if (!response.ok) {
        // Keep default same-origin API routes when config is not available.
        return;
    }
    const config = await response.json();
    apiUrl = config.API_URL || apiUrl;
}

function appendMessage(sender, text) {
    const chatDisplay = document.getElementById('chatDisplay');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    if (sender === 'model') {
        const textSpan = document.createElement('span');
        textSpan.textContent = text;

        const thumbsContainer = document.createElement('div');
        thumbsContainer.className = 'thumbs';

        const thumbsUp = document.createElement('span');
        thumbsUp.textContent = '👍';
        thumbsUp.className = 'thumb-icon';

        const thumbsDown = document.createElement('span');
        thumbsDown.textContent = '👎';
        thumbsDown.className = 'thumb-icon';

        thumbsUp.addEventListener('click', () => handleThumbClick(text, 'thumbsUp'));
        thumbsDown.addEventListener('click', () => handleThumbClick(text, 'thumbsDown'));

        thumbsContainer.appendChild(thumbsUp);
        thumbsContainer.appendChild(thumbsDown);

        messageDiv.appendChild(textSpan);
        messageDiv.appendChild(thumbsContainer);
    } else {
        messageDiv.textContent = text;
    }

    chatDisplay.appendChild(messageDiv);
    chatDisplay.scrollTop = chatDisplay.scrollHeight;
    pushConversationEntry(sender, text);
}

async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const query = userInput.value.trim();

    if (!query) return;

    appendMessage('user', query);
    userInput.value = '';

    try {
        const response = await fetch(apiUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query,
                conversation_context: buildConversationContextPayload(),
            }),
        });

        if (response.ok) {
            const data = await response.json();
            appendMessage('model', data.response || 'Erro: Resposta inválida.');
            try {
                await loadContextState();
                if (!window.EventSource) {
                    await loadMemoryGraph({ silent: true });
                }
            } catch (refreshError) {
                console.error('Failed to refresh context/memory state after chat.', refreshError);
            }
        } else {
            const errorBody = await response.json().catch(() => ({}));
            const errorMessage = errorBody.error || `Erro da API (status ${response.status}).`;
            appendMessage('model', errorMessage);
        }
    } catch (error) {
        appendMessage('model', 'Erro: Não foi possível completar a requisição.');
    }
}

async function handleThumbClick(messageContent, thumbType) {
    const feedback = {
        feedback_type: thumbType,
        message: messageContent
    };

    try {
        const response = await fetch(`${dataApiBaseUrl}/feedback`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(feedback),
        });

        if (response.ok) {
            console.log('Feedback sent successfully');
        } else {
            const errorBody = await response.json().catch(() => ({}));
            console.error('Error sending feedback', response.status, errorBody);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

function formatBytes(bytes) {
    const value = Number(bytes || 0);
    if (value < 1024) {
        return `${value} B`;
    }
    const units = ['KB', 'MB', 'GB'];
    let unitIndex = -1;
    let size = value;
    while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex += 1;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
}

function blockedReasonMessage(reasons) {
    if (!reasons || reasons.length === 0) {
        return '';
    }
    if (reasons.includes('document_limit')) {
        return 'Maximum number of files reached.';
    }
    if (reasons.includes('total_size_limit')) {
        return 'Maximum total size reached.';
    }
    return 'Upload is currently unavailable.';
}

function renderContextState() {
    const documents = contextState.documents || [];
    const limits = contextState.limits || {};
    const usage = contextState.usage || {};

    const maxDocuments = Number(limits.max_documents || 3);
    const currentDocuments = Number(usage.document_count || documents.length || 0);

    const filesCounter = document.getElementById('filesCounter');
    const contextStatusIcon = document.getElementById('contextStatusIcon');
    const contextDocsCount = document.getElementById('contextDocsCount');
    const documentsList = document.getElementById('documentsList');
    const uploadContainer = document.getElementById('uploadContainer');

    filesCounter.textContent = `Context (${currentDocuments}/${maxDocuments})`;
    if (contextStatusIcon) {
        contextStatusIcon.textContent = getContextStatusIcon(currentDocuments);
    }
    if (contextDocsCount) {
        contextDocsCount.textContent = String(currentDocuments);
    }

    documentsList.innerHTML = '';
    documents.forEach((doc) => {
        const item = document.createElement('div');
        item.className = 'document-item';

        const info = document.createElement('div');
        info.className = 'document-info';

        const title = document.createElement('div');
        title.className = 'document-title';

        const icon = document.createElement('span');
        icon.className = 'document-icon';
        icon.textContent = getFileTypeIcon(doc.file_type);

        const name = document.createElement('div');
        name.className = 'document-name';
        name.textContent = doc.filename;

        title.appendChild(icon);
        title.appendChild(name);

        const meta = document.createElement('div');
        meta.className = 'document-meta';
        const detail = getDocumentDetail(doc);
        const sizeLabel = formatBytes(doc.size_bytes || 0);
        meta.textContent = `${detail} | ${sizeLabel}`;

        info.appendChild(title);
        info.appendChild(meta);

        const removeButton = document.createElement('button');
        removeButton.className = 'remove-file-button';
        removeButton.type = 'button';
        removeButton.textContent = 'X';
        removeButton.title = 'Remove file';
        removeButton.addEventListener('click', () => deleteDocument(doc.id));

        item.appendChild(info);
        item.appendChild(removeButton);
        documentsList.appendChild(item);
    });

    uploadContainer.style.display = contextState.is_upload_blocked ? 'none' : 'block';
}

function isInternalMemoryNodeId(value) {
    const normalized = String(value || '').trim().toLowerCase();
    return normalized ? INTERNAL_MEMORY_NODE_IDS.has(normalized) : false;
}

function normalizeMemoryGraphResponse(payload) {
    const visualization = payload && typeof payload === 'object' ? (payload.visualization || {}) : {};
    const meta = payload && typeof payload === 'object' ? (payload.meta || {}) : {};
    const nodes = Array.isArray(visualization.nodes) ? visualization.nodes : [];
    const edges = Array.isArray(visualization.edges) ? visualization.edges : [];
    const normalizedNodes = nodes.map((node) => ({
        id: String(node.id || ''),
        label: String(node.label || node.id || ''),
        type: String(node.type || 'unknown'),
        observation_count: Number(node.observation_count || 0)
    })).filter((node) => node.id && !isInternalMemoryNodeId(node.id));
    const visibleNodeIds = new Set(normalizedNodes.map((node) => node.id));
    const normalizedEdges = edges.map((edge) => ({
        id: String(edge.id || ''),
        source: String(edge.source || ''),
        target: String(edge.target || ''),
        label: String(edge.label || 'related_to')
    })).filter((edge) => (
        edge.source
        && edge.target
        && !isInternalMemoryNodeId(edge.source)
        && !isInternalMemoryNodeId(edge.target)
        && visibleNodeIds.has(edge.source)
        && visibleNodeIds.has(edge.target)
    ));
    return {
        nodes: normalizedNodes,
        edges: normalizedEdges,
        meta: {
            entity_count: normalizedNodes.length,
            relation_count: normalizedEdges.length,
            updated_at: String(meta.updated_at || '')
        }
    };
}

function colorForNodeType(type) {
    const palette = ['#2c7be5', '#20c997', '#f59f00', '#e64980', '#845ef7', '#228be6'];
    const key = String(type || 'unknown');
    let hash = 0;
    for (let i = 0; i < key.length; i += 1) {
        hash = ((hash << 5) - hash) + key.charCodeAt(i);
        hash |= 0;
    }
    return palette[Math.abs(hash) % palette.length];
}

function destroyMemoryNetwork() {
    if (memoryNetwork) {
        memoryNetwork.destroy();
        memoryNetwork = null;
    }
}

function toMemoryNetworkNodes(nodes) {
    return nodes.map((node) => {
        const nodeColor = colorForNodeType(node.type);
        return {
            id: node.id,
            label: node.label,
            title: `${node.label}\nType: ${node.type}\nObservations: ${node.observation_count}`,
            shape: 'dot',
            color: {
                background: nodeColor,
                border: '#ffffff',
                highlight: { background: nodeColor, border: '#1f2a37' }
            },
            value: 18 + Math.min(14, Math.max(0, Number(node.observation_count || 0))),
            font: {
                color: '#1f2a37',
                size: 12,
                face: 'Arial'
            }
        };
    });
}

function toMemoryNetworkEdges(edges) {
    return edges.map((edge, index) => ({
        id: edge.id || `${edge.source}-${edge.target}-${edge.label}-${index}`,
        from: edge.source,
        to: edge.target,
        label: edge.label,
        arrows: 'to',
        color: { color: '#98a6b3', highlight: '#2c7be5' },
        font: { size: 10, align: 'middle', color: '#495057' },
        smooth: { type: 'dynamic', roundness: 0.2 }
    }));
}

function renderMemoryGraphSvg(nodes, edges) {
    const canvas = document.getElementById('memoryGraphCanvas');
    if (!canvas) {
        return;
    }

    if (!nodes.length) {
        destroyMemoryNetwork();
        canvas.innerHTML = '';
        const empty = document.createElement('div');
        empty.className = 'memory-graph-empty';
        empty.textContent = memoryUiStatus === 'loading'
            ? 'Loading knowledge graph...'
            : 'Knowledge graph is empty. Ask the assistant to remember facts to populate it.';
        canvas.appendChild(empty);
        return;
    }

    if (!window.vis || !window.vis.Network) {
        destroyMemoryNetwork();
        canvas.innerHTML = '';
        const fallback = document.createElement('div');
        fallback.className = 'memory-graph-empty';
        fallback.textContent = 'Interactive graph library unavailable. Please reload the page.';
        canvas.appendChild(fallback);
        return;
    }

    const graphData = {
        nodes: new window.vis.DataSet(toMemoryNetworkNodes(nodes)),
        edges: new window.vis.DataSet(toMemoryNetworkEdges(edges))
    };
    const graphOptions = {
        autoResize: true,
        interaction: {
            dragNodes: true,
            dragView: true,
            zoomView: true,
            hover: true,
            multiselect: true,
            navigationButtons: false,
            keyboard: { enabled: true, bindToWindow: false }
        },
        layout: {
            improvedLayout: true,
            randomSeed: 17
        },
        physics: {
            enabled: true,
            stabilization: { enabled: true, iterations: 180, fit: true },
            barnesHut: {
                gravitationalConstant: -6500,
                springLength: 140,
                springConstant: 0.04,
                damping: 0.24
            }
        },
        nodes: {
            borderWidth: 2,
            scaling: { min: 14, max: 36 }
        },
        edges: {
            width: 1.8
        }
    };

    if (!memoryNetwork) {
        canvas.innerHTML = '';
        memoryNetwork = new window.vis.Network(canvas, graphData, graphOptions);
    } else {
        memoryNetwork.setData(graphData);
        memoryNetwork.setOptions(graphOptions);
    }
}

function renderMemoryLegend(nodes, edges, meta) {
    const legend = document.getElementById('memoryGraphLegend');
    if (!legend) {
        return;
    }
    const typeCount = {};
    nodes.forEach((node) => {
        const key = String(node.type || 'unknown');
        typeCount[key] = (typeCount[key] || 0) + 1;
    });
    const topTypes = Object.entries(typeCount)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
        .map(([type, count]) => `${type}: ${count}`);
    const updatedAt = meta.updated_at ? new Date(meta.updated_at).toLocaleTimeString() : '--';
    legend.textContent = `Nodes: ${nodes.length} | Edges: ${edges.length} | Top types: ${topTypes.join(', ') || 'none'}`;
}

function renderMemoryGraphState() {
    const nodes = memoryGraphState.nodes || [];
    const edges = memoryGraphState.edges || [];
    const meta = memoryGraphState.meta || {};

    const counter = document.getElementById('memoryCounter');
    const statusIcon = document.getElementById('memoryStatusIcon');
    const nodesCount = document.getElementById('memoryNodesCount');

    if (counter) {
        counter.textContent = 'Knowledge Graph';
    }
    if (statusIcon) {
        statusIcon.textContent = getMemoryStatusIcon(nodes.length);
    }
    if (nodesCount) {
        nodesCount.textContent = String(nodes.length);
    }

    renderMemoryGraphSvg(nodes, edges);
    renderMemoryLegend(nodes, edges, meta);
}

function handleMemoryGraphStreamEvent(event) {
    if (!event || typeof event.data !== 'string' || !event.data.trim()) {
        return;
    }
    try {
        const payload = JSON.parse(event.data);
        memoryGraphState = normalizeMemoryGraphResponse(payload);
        memoryUiStatus = memoryGraphState.nodes.length > 0 ? 'ready' : 'idle';
        renderMemoryGraphState();
    } catch (error) {
        console.error('Failed to parse memory graph SSE payload.', error);
    }
}

function startMemoryGraphStream() {
    if (!window.EventSource || memoryPanelCollapsed || memoryGraphEventSource) {
        return;
    }
    const streamUrl = `${dataApiBaseUrl}/memory/graph/events`;
    memoryGraphEventSource = new EventSource(streamUrl);

    memoryGraphEventSource.addEventListener('snapshot', handleMemoryGraphStreamEvent);
    memoryGraphEventSource.addEventListener('update', handleMemoryGraphStreamEvent);
    memoryGraphEventSource.onmessage = handleMemoryGraphStreamEvent;
    memoryGraphEventSource.onerror = () => {
        if (!memoryPanelCollapsed) {
            console.warn('Memory graph SSE connection interrupted. Waiting for auto-reconnect.');
        }
    };
}

function stopMemoryGraphStream() {
    if (!memoryGraphEventSource) {
        return;
    }
    memoryGraphEventSource.close();
    memoryGraphEventSource = null;
}

async function loadMemoryGraph(options = {}) {
    if (memoryRequestInFlight) {
        return;
    }
    memoryRequestInFlight = true;

    const silent = Boolean(options.silent);
    if (!silent) {
        memoryUiStatus = 'loading';
        renderMemoryGraphState();
    }

    try {
        const response = await fetch(`${dataApiBaseUrl}/memory/graph`);
        if (!response.ok) {
            throw new Error('Failed to load memory graph.');
        }
        const data = await response.json();
        memoryGraphState = normalizeMemoryGraphResponse(data);
        memoryUiStatus = memoryGraphState.nodes.length > 0 ? 'ready' : 'idle';
    } catch (error) {
        console.error(error);
        memoryGraphState = {
            nodes: [],
            edges: [],
            meta: { entity_count: 0, relation_count: 0, updated_at: '' }
        };
        memoryUiStatus = 'error';
    } finally {
        memoryRequestInFlight = false;
    }

    renderMemoryGraphState();
}

async function loadContextState() {
    contextUiStatus = 'loading';
    renderContextState();

    try {
        const response = await fetch(`${dataApiBaseUrl}/documents`);
        if (!response.ok) {
            throw new Error('Failed to load documents state.');
        }
        const data = await response.json();
        contextState = data;
        contextUiStatus = Number(data?.usage?.document_count || 0) > 0 ? 'ready' : 'idle';
    } catch (error) {
        console.error(error);
        contextState = {
            documents: [],
            limits: { max_documents: 3 },
            usage: { document_count: 0, total_size_bytes: 0 },
            is_upload_blocked: false,
            blocked_reasons: []
        };
        contextUiStatus = 'error';
    }
    renderContextState();
}

async function deleteDocument(documentId) {
    if (!documentId) {
        return;
    }

    contextUiStatus = 'loading';
    renderContextState();
    setLoadingOverlayVisible(true);

    try {
        const response = await fetch(`${dataApiBaseUrl}/documents/${documentId}`, {
            method: 'DELETE'
        });
        const body = await response.json().catch(() => ({}));

        if (!response.ok) {
            alert(body.error || 'Failed to remove document.');
            contextUiStatus = 'error';
            renderContextState();
            return;
        }

        if (body.context) {
            contextState = body.context;
            contextUiStatus = Number(contextState?.usage?.document_count || 0) > 0 ? 'ready' : 'idle';
            renderContextState();
        } else {
            await loadContextState();
        }
    } catch (error) {
        console.error(error);
        alert('Failed to remove document.');
        contextUiStatus = 'error';
        renderContextState();
    } finally {
        setLoadingOverlayVisible(false);
    }
}

async function handleDrop(event) {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
        await processFile(file);
    }
}

function allowDrag(event) {
    event.preventDefault();
}

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (file) {
        await processFile(file);
    }
    fileInput.value = '';
}

async function processFile(file) {
    if (contextState.is_upload_blocked) {
        alert(blockedReasonMessage(contextState.blocked_reasons) || 'Upload is currently blocked.');
        return;
    }

    contextUiStatus = 'loading';
    renderContextState();
    const formData = new FormData();
    formData.append('file', file);

    setLoadingOverlayVisible(true);

    try {
        const response = await fetch(`${dataApiBaseUrl}/upload`, {
            method: 'POST',
            body: formData
        });

        const body = await response.json().catch(() => ({}));

        if (!response.ok) {
            alert(body.error || 'Error processing file');
            contextUiStatus = 'error';
            await loadContextState();
            return;
        }

        if (body.context) {
            contextState = body.context;
            contextUiStatus = Number(contextState?.usage?.document_count || 0) > 0 ? 'ready' : 'idle';
            renderContextState();
        } else {
            await loadContextState();
        }

        alert(body.message || 'File uploaded and processed successfully!');
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to upload file');
        contextUiStatus = 'error';
        await loadContextState();
    } finally {
        setLoadingOverlayVisible(false);
    }
}

document.getElementById('userInput').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

(async function initializeApp() {
    const toggleButton = document.getElementById('contextToggleButton');
    const collapsedRail = document.getElementById('contextCollapsedRail');
    const chatCollapsedRail = document.getElementById('chatCollapsedRail');
    const memoryToggleButton = document.getElementById('memoryToggleButton');
    const memoryFocusButton = document.getElementById('memoryFocusButton');
    const memoryCollapsedRail = document.getElementById('memoryCollapsedRail');
    if (toggleButton) {
        toggleButton.addEventListener('click', toggleContextPanel);
    }
    if (collapsedRail) {
        collapsedRail.addEventListener('click', () => setContextPanelCollapsed(false));
    }
    if (chatCollapsedRail) {
        chatCollapsedRail.addEventListener('click', () => setChatPanelCollapsed(false));
    }
    if (memoryToggleButton) {
        memoryToggleButton.addEventListener('click', toggleMemoryPanel);
    }
    if (memoryFocusButton) {
        memoryFocusButton.addEventListener('click', toggleMemoryFocusMode);
    }
    if (memoryCollapsedRail) {
        memoryCollapsedRail.addEventListener('click', () => setMemoryPanelCollapsed(false));
    }

    window.addEventListener('resize', () => {
        if (window.innerWidth <= 900 && memoryFocusMode) {
            setMemoryFocusMode(false);
        }
        if (window.innerWidth > 900 && memoryNetwork) {
            memoryNetwork.redraw();
        }
    });
    window.addEventListener('beforeunload', () => {
        stopMemoryGraphStream();
        destroyMemoryNetwork();
    });

    restoreMemoryPanelWidth();
    initializeMemoryPanelResize();
    setContextPanelCollapsed(true);
    setChatPanelCollapsed(false, { skipFocusReset: true });
    setMemoryPanelCollapsed(true);
    renderMemoryGraphState();

    await getConfig();
    await loadContextState();
    await loadMemoryGraph({ silent: true });
})();
