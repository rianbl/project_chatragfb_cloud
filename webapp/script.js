let apiUrl = 'http://localhost:8081/chat';
const dataApiBaseUrl = 'http://localhost:5001';

let contextState = {
    documents: [],
    limits: { max_documents: 3, max_total_size_bytes: 0 },
    usage: { document_count: 0, total_size_bytes: 0 },
    is_upload_blocked: false,
    blocked_reasons: []
};

function setLoadingOverlayVisible(isVisible) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (!loadingOverlay) {
        return;
    }
    loadingOverlay.style.display = isVisible ? 'flex' : 'none';
}

async function getConfig() {
    const response = await fetch('config.json');
    if (!response.ok) {
        console.error('Failed to load configuration');
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
            body: JSON.stringify({ query }),
        });

        if (response.ok) {
            const data = await response.json();
            appendMessage('model', data.response || 'Erro: Resposta inválida.');
        } else {
            appendMessage('model', 'Erro ao se conectar com a API.');
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
        const response = await fetch('http://localhost:5002/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(feedback),
        });

        if (response.ok) {
            console.log('Feedback sent successfully');
        } else {
            console.error('Error sending feedback');
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
    const documentsList = document.getElementById('documentsList');
    const uploadContainer = document.getElementById('uploadContainer');

    filesCounter.textContent = `Context (${currentDocuments}/${maxDocuments})`;

    documentsList.innerHTML = '';
    documents.forEach((doc) => {
        const item = document.createElement('div');
        item.className = 'document-item';

        const info = document.createElement('div');
        info.className = 'document-info';

        const name = document.createElement('div');
        name.className = 'document-name';
        name.textContent = doc.filename;

        const meta = document.createElement('div');
        meta.className = 'document-meta';
        const chunks = Number(doc.chunk_count || 0);
        const sizeLabel = formatBytes(doc.size_bytes || 0);
        meta.textContent = `${(doc.file_type || '').toUpperCase()} | ${chunks} chunks | ${sizeLabel}`;

        info.appendChild(name);
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
    try {
        const response = await fetch(`${dataApiBaseUrl}/documents`);
        if (!response.ok) {
            throw new Error('Failed to load documents state.');
        }
        const data = await response.json();
        contextState = data;
    } catch (error) {
        console.error(error);
        contextState = {
            documents: [],
            limits: { max_documents: 3 },
            usage: { document_count: 0, total_size_bytes: 0 },
            is_upload_blocked: false,
            blocked_reasons: []
        };
    }
    renderContextState();
}

async function deleteDocument(documentId) {
    if (!documentId) {
        return;
    }

    setLoadingOverlayVisible(true);

    try {
        const response = await fetch(`${dataApiBaseUrl}/documents/${documentId}`, {
            method: 'DELETE'
        });
        const body = await response.json().catch(() => ({}));

        if (!response.ok) {
            alert(body.error || 'Failed to remove document.');
            return;
        }

        if (body.context) {
            contextState = body.context;
            renderContextState();
        } else {
            await loadContextState();
        }
    } catch (error) {
        console.error(error);
        alert('Failed to remove document.');
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
            await loadContextState();
            return;
        }

        if (body.context) {
            contextState = body.context;
            renderContextState();
        } else {
            await loadContextState();
        }

        alert(body.message || 'File uploaded and processed successfully!');
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to upload file');
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
    await getConfig();
    await loadContextState();
})();
