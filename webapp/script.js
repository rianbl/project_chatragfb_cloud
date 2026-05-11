let apiUrl = 'http://localhost:8081/chat'; // Default API URL

async function getConfig() {
    const response = await fetch('config.json');
    if (!response.ok) {
        console.error('Failed to load configuration');
        return; // Keep default apiUrl
    }
    const config = await response.json();
    apiUrl = config.API_URL || apiUrl; // Update apiUrl if found in JSON
}

// Call getConfig to fetch the configuration on page load
getConfig();

function appendMessage(sender, text) {
    const chatDisplay = document.getElementById('chatDisplay');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;

    if (sender === 'model') {
        // Add text and icons for the model's message
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
        // User message: only text
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
    const file = document.getElementById('fileInput').files[0];
    if (file) {
        await processFile(file);
    }
}

async function processFile(file) {
    const formData = new FormData();
    formData.append('file', file);

    const fileNameDisplay = document.getElementById('fileNameDisplay');
    fileNameDisplay.style.fontWeight = "bold";
    fileNameDisplay.textContent = file.name;

    const loadingOverlay = document.getElementById('loadingOverlay');
    loadingOverlay.style.display = 'flex';

    try {
        const response = await fetch('http://localhost:5001/upload', {
            method: 'POST',
            body: formData
        });
        if (response.ok) {
            alert('File uploaded and processed successfully!');
        } else {
            alert('Error processing file');
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to upload file');
    } finally {
        loadingOverlay.style.display = 'none';
    }
}

document.getElementById('userInput').addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});
