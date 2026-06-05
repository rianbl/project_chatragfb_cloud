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
let memoryPanelCollapsed = false;
let memoryUiStatus = 'idle';
let memoryGraphEventSource = null;
let memoryRequestInFlight = false;
const MAX_CONVERSATION_CONTEXT_MESSAGES = 12;
let conversationHistory = [];
let memoryGraphState = {
    nodes: [],
    edges: [],
    meta: { entity_count: 0, relation_count: 0, updated_at: '' }
};

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

function setMemoryPanelCollapsed(isCollapsed) {
    const memoryPanel = document.getElementById('memoryPanel');
    const toggleButton = document.getElementById('memoryToggleButton');
    if (!memoryPanel || !toggleButton) {
        return;
    }

    memoryPanelCollapsed = Boolean(isCollapsed);
    memoryPanel.classList.toggle('is-collapsed', memoryPanelCollapsed);
    toggleButton.setAttribute('aria-expanded', String(!memoryPanelCollapsed));
    toggleButton.textContent = memoryPanelCollapsed ? '▶' : '◀';
    toggleButton.title = memoryPanelCollapsed ? 'Expand memory graph panel' : 'Collapse memory graph panel';

    if (memoryPanelCollapsed) {
        stopMemoryGraphStream();
    } else {
        startMemoryGraphStream();
        if (!window.EventSource) {
            loadMemoryGraph({ silent: true }).catch((error) => console.error(error));
        }
    }
}

function toggleMemoryPanel() {
    setMemoryPanelCollapsed(!memoryPanelCollapsed);
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

function normalizeMemoryGraphResponse(payload) {
    const visualization = payload && typeof payload === 'object' ? (payload.visualization || {}) : {};
    const meta = payload && typeof payload === 'object' ? (payload.meta || {}) : {};
    const nodes = Array.isArray(visualization.nodes) ? visualization.nodes : [];
    const edges = Array.isArray(visualization.edges) ? visualization.edges : [];
    return {
        nodes: nodes.map((node) => ({
            id: String(node.id || ''),
            label: String(node.label || node.id || ''),
            type: String(node.type || 'unknown'),
            observation_count: Number(node.observation_count || 0)
        })).filter((node) => node.id),
        edges: edges.map((edge) => ({
            id: String(edge.id || ''),
            source: String(edge.source || ''),
            target: String(edge.target || ''),
            label: String(edge.label || 'related_to')
        })).filter((edge) => edge.source && edge.target),
        meta: {
            entity_count: Number(meta.entity_count || 0),
            relation_count: Number(meta.relation_count || 0),
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

function renderMemoryGraphSvg(nodes, edges) {
    const canvas = document.getElementById('memoryGraphCanvas');
    if (!canvas) {
        return;
    }

    canvas.innerHTML = '';

    if (!nodes.length) {
        const empty = document.createElement('div');
        empty.className = 'memory-graph-empty';
        empty.textContent = memoryUiStatus === 'loading'
            ? 'Loading memory graph...'
            : 'Memory graph is empty. Ask the assistant to remember facts to populate it.';
        canvas.appendChild(empty);
        return;
    }

    const width = Math.max(canvas.clientWidth, 220);
    const height = Math.max(canvas.clientHeight, 320);
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');

    const defs = document.createElementNS(ns, 'defs');
    const marker = document.createElementNS(ns, 'marker');
    marker.setAttribute('id', 'memoryArrow');
    marker.setAttribute('markerWidth', '10');
    marker.setAttribute('markerHeight', '10');
    marker.setAttribute('refX', '8');
    marker.setAttribute('refY', '3');
    marker.setAttribute('orient', 'auto');
    const arrowPath = document.createElementNS(ns, 'path');
    arrowPath.setAttribute('d', 'M0,0 L0,6 L9,3 z');
    arrowPath.setAttribute('fill', '#6c757d');
    marker.appendChild(arrowPath);
    defs.appendChild(marker);
    svg.appendChild(defs);

    const centerX = width / 2;
    const centerY = height / 2;
    const radius = Math.max(70, Math.min(width, height) / 2 - 46);
    const positions = new Map();

    nodes.forEach((node, index) => {
        const angle = (Math.PI * 2 * index) / nodes.length;
        const x = nodes.length === 1 ? centerX : centerX + radius * Math.cos(angle);
        const y = nodes.length === 1 ? centerY : centerY + radius * Math.sin(angle);
        positions.set(node.id, { x, y, node });
    });

    edges.forEach((edge) => {
        const source = positions.get(edge.source);
        const target = positions.get(edge.target);
        if (!source || !target) {
            return;
        }

        const line = document.createElementNS(ns, 'line');
        line.setAttribute('x1', String(source.x));
        line.setAttribute('y1', String(source.y));
        line.setAttribute('x2', String(target.x));
        line.setAttribute('y2', String(target.y));
        line.setAttribute('stroke', '#98a6b3');
        line.setAttribute('stroke-width', '1.8');
        line.setAttribute('marker-end', 'url(#memoryArrow)');
        svg.appendChild(line);

        const label = document.createElementNS(ns, 'text');
        label.setAttribute('x', String((source.x + target.x) / 2));
        label.setAttribute('y', String((source.y + target.y) / 2 - 4));
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('font-size', '10');
        label.setAttribute('fill', '#495057');
        label.textContent = edge.label;
        svg.appendChild(label);
    });

    nodes.forEach((node) => {
        const position = positions.get(node.id);
        if (!position) {
            return;
        }
        const group = document.createElementNS(ns, 'g');

        const circle = document.createElementNS(ns, 'circle');
        circle.setAttribute('cx', String(position.x));
        circle.setAttribute('cy', String(position.y));
        circle.setAttribute('r', '20');
        circle.setAttribute('fill', colorForNodeType(node.type));
        circle.setAttribute('fill-opacity', '0.9');
        circle.setAttribute('stroke', '#ffffff');
        circle.setAttribute('stroke-width', '2');
        group.appendChild(circle);

        const text = document.createElementNS(ns, 'text');
        text.setAttribute('x', String(position.x));
        text.setAttribute('y', String(position.y + 35));
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('font-size', '11');
        text.setAttribute('fill', '#212529');
        text.textContent = node.label.length > 18 ? `${node.label.slice(0, 18)}...` : node.label;
        group.appendChild(text);

        svg.appendChild(group);
    });

    canvas.appendChild(svg);
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
        counter.textContent = 'Memory Graph';
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
    const memoryToggleButton = document.getElementById('memoryToggleButton');
    const memoryCollapsedRail = document.getElementById('memoryCollapsedRail');
    if (toggleButton) {
        toggleButton.addEventListener('click', toggleContextPanel);
    }
    if (collapsedRail) {
        collapsedRail.addEventListener('click', () => setContextPanelCollapsed(false));
    }
    if (memoryToggleButton) {
        memoryToggleButton.addEventListener('click', toggleMemoryPanel);
    }
    if (memoryCollapsedRail) {
        memoryCollapsedRail.addEventListener('click', () => setMemoryPanelCollapsed(false));
    }

    window.addEventListener('resize', () => {
        renderMemoryGraphState();
    });
    window.addEventListener('beforeunload', () => {
        stopMemoryGraphStream();
    });

    setContextPanelCollapsed(true);
    setMemoryPanelCollapsed(true);
    renderMemoryGraphState();

    await getConfig();
    await loadContextState();
    await loadMemoryGraph({ silent: true });
})();
