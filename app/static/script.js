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
const MAX_CONVERSATION_CONTEXT_MESSAGES = 12;
let conversationHistory = [];

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
            } catch (refreshError) {
                console.error('Failed to refresh context state after chat.', refreshError);
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
    if (toggleButton) {
        toggleButton.addEventListener('click', toggleContextPanel);
    }
    if (collapsedRail) {
        collapsedRail.addEventListener('click', () => setContextPanelCollapsed(false));
    }
    setContextPanelCollapsed(true);

    await getConfig();
    await loadContextState();
})();
