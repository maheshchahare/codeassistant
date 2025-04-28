// Global variables
const chatMessages = document.getElementById('chat-messages');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');
const modelSelect = document.getElementById('model-select');
const languageSelect = document.getElementById('language-select');
const taskSelect = document.getElementById('task-select');
const clearChatButton = document.getElementById('clear-chat');
const checkHealthButton = document.getElementById('check-health');
const toggleSettingsButton = document.getElementById('toggle-settings');
const settingsPanel = document.getElementById('settings-panel');
const overlay = document.getElementById('overlay');
const sessionList = document.getElementById('session-list');
const createSessionButton = document.getElementById('create-session-btn');
const sessionsSidebar = document.querySelector('.sessions-sidebar');

// Initialize session management
let currentSessionId = localStorage.getItem('currentSessionId') || null;
let sessions = [];

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Initialize the first session if none exists
    initializeSessions();
    
    sendButton.addEventListener('click', handleSendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.ctrlKey && e.key === 'Enter') {
            handleSendMessage();
        }
    });
    
    clearChatButton.addEventListener('click', clearChat);
    checkHealthButton.addEventListener('click', checkHealth);
    toggleSettingsButton.addEventListener('click', toggleSettings);
    overlay.addEventListener('click', () => {
        closeSettings();
        closeSidebar();
    });
    createSessionButton.addEventListener('click', createNewSession);

    // Auto-resize textarea
    userInput.addEventListener('input', autoResizeTextarea);
    
    // Fetch available models
    fetchAvailableModels();
    
    // Add mobile toggle for sidebar
    const toggleSidebarBtn = document.createElement('button');
    toggleSidebarBtn.id = 'toggle-sidebar';
    toggleSidebarBtn.className = 'toggle-sidebar';
    toggleSidebarBtn.innerHTML = '<i class="fas fa-bars"></i>';
    toggleSidebarBtn.setAttribute('aria-label', 'Toggle sidebar');
    
    document.querySelector('.header').prepend(toggleSidebarBtn);
    
    toggleSidebarBtn.addEventListener('click', toggleSidebar);
});

// Call this function after loading sessions
function initializeSessions() {
    // Load existing sessions from localStorage
    sessions = JSON.parse(localStorage.getItem('sessions')) || [];
    
    // If no sessions exist, create a default one
    if (sessions.length === 0) {
        createNewSession();
    } else {
        // Load existing sessions
        loadSessions();
        
        // Set current session
        if (!currentSessionId || !sessions.some(session => session.id === currentSessionId)) {
            currentSessionId = sessions[0].id;
            localStorage.setItem('currentSessionId', currentSessionId);
        }
        
        // Load messages for current session
        loadSessionMessages(currentSessionId);
        
        // Mark current session as active
        updateActiveSession();
        
        // Setup hover effects
        setupSessionHoverEffects();
    }
}

function loadSessions() {
    sessionList.innerHTML = '';  // Clear existing session list

    sessions.forEach(session => {
        const sessionItem = document.createElement('li');
        sessionItem.dataset.id = session.id;
        sessionItem.className = 'session-item';
        if (session.id === currentSessionId) {
            sessionItem.classList.add('active');
        }
        
        const icon = document.createElement('i');
        // Use chat bubble icon for better visual representation
        icon.className = 'fas fa-comment-dots';
        
        const sessionName = document.createElement('span');
        sessionName.textContent = session.name;
        
        sessionItem.appendChild(icon);
        sessionItem.appendChild(sessionName);
        
        sessionItem.addEventListener('click', () => switchSession(session.id));
        sessionList.appendChild(sessionItem);
    });
}

// This function handles the hover effect for session items
function setupSessionHoverEffects() {
    // Add event listeners for all session items
    document.querySelectorAll('.session-item').forEach(item => {
        item.addEventListener('mouseenter', () => {
            if (!item.classList.contains('active')) {
                item.style.backgroundColor = '#f0f4ff';
            }
        });
        
        item.addEventListener('mouseleave', () => {
            if (!item.classList.contains('active')) {
                item.style.backgroundColor = '';
            }
        });
    });
}

async function createNewSession() {
    try {
        // Call backend to create new session
        const response = await fetch('/sessions', { method: 'POST' });
        const data = await response.json();

        // Backend returns { id, name }
        const newSessionId = data.id;
        const sessionName = data.name;

        const session = { 
            id: newSessionId, 
            name: sessionName,
            messages: []
        };

        sessions.unshift(session);  // Add to beginning
        localStorage.setItem('sessions', JSON.stringify(sessions));
        currentSessionId = newSessionId;
        localStorage.setItem('currentSessionId', currentSessionId);

        loadSessions();
        clearChat();
        // addBotMessage(`Welcome to a new chat session!`);
        closeSidebar();

    } catch (error) {
        console.error('Error creating session:', error);
        addBotMessage(`Failed to create a new session. Please try again.`);
    }
}

function switchSession(sessionId) {
    if (sessionId === currentSessionId) return;
    
    // Save current messages if needed
    saveSessionMessages(currentSessionId);
    
    // Switch to selected session
    currentSessionId = sessionId;
    localStorage.setItem('currentSessionId', currentSessionId);
    
    // Clear chat and load session messages
    clearChatSilent();
    loadSessionMessages(sessionId);
    
    // Update active session in UI
    updateActiveSession();
    
    // Close sidebar on mobile after switching
    if (window.innerWidth <= 768) {
        closeSidebar();
    }
}

function updateActiveSession() {
    // Remove active class from all sessions
    document.querySelectorAll('.session-item').forEach(item => {
        item.classList.remove('active');
    });
    
    // Add active class to current session
    const activeItem = document.querySelector(`.session-item[data-id="${currentSessionId}"]`);
    if (activeItem) {
        activeItem.classList.add('active');
    }
}

function saveSessionMessages(sessionId) {
    // Find the session in our array
    const sessionIndex = sessions.findIndex(s => s.id === sessionId);
    if (sessionIndex === -1) return;
    
    // Get all messages from the chat and convert to storable format
    const messageElements = chatMessages.querySelectorAll('.message');
    const messages = Array.from(messageElements).map(el => {
        const isUser = el.classList.contains('user-message');
        const content = el.querySelector('.message-content').innerHTML;
        return { content, isUser };
    });
    
    // Update session with messages
    sessions[sessionIndex].messages = messages;
    
    // Save to localStorage
    localStorage.setItem('sessions', JSON.stringify(sessions));
}

function loadSessionMessages(sessionId) {
    // Find the session
    const session = sessions.find(s => s.id === sessionId);
    if (!session || !session.messages) return;
    
    // Add each message to the chat
    session.messages.forEach(msg => {
        if (msg.isUser) {
            addUserMessageRaw(msg.content);
        } else {
            addBotMessageRaw(msg.content);
        }
    });
}

// UI toggle functions
function toggleSidebar() {
    sessionsSidebar.classList.toggle('show');
    overlay.classList.toggle('show');
}

function closeSidebar() {
    sessionsSidebar.classList.remove('show');
    overlay.classList.remove('show');
}

function toggleSettings() {
    settingsPanel.classList.toggle('show');
    overlay.classList.toggle('show');
}

function closeSettings() {
    settingsPanel.classList.remove('show');
    overlay.classList.remove('show');
}

// Format and display response
function displayQueryResponse(data) {
    const { response, tone } = data;
    let formattedResponse = '';

    if (tone === 'conversation') {
        addBotMessage(escapeHtml(response));
        return;
    }

    // formattedResponse += '<h3>Explanation</h3>';
    formattedResponse += `<div class="explanation">${escapeHtml(response)}</div>`;

    addBotMessage(formattedResponse);
}



// Send Message Function
async function handleSendMessage() {
    const message = userInput.value.trim();
    if (!message) return;
    console.log("Entered handleSendMessage")
    // Display user message
    addUserMessage(message);

    // Clear input
    userInput.value = '';
    userInput.style.height = 'auto'; // reset height

    // Show loading
    const loadingId = addLoadingMessage();
    console.log("Executed addLoadingMessage")
    try {
        const requestData = {
            query: message,
            language: languageSelect.value,
            task: taskSelect.value,
            model: modelSelect.value,
            sessionId: currentSessionId,
            includeHistory: true,  // Add this line
            historyLimit: 10       // Add this line
        };
        console.log("Executed requestData")
        const response = await fetch('/query', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(requestData)
        });
        console.log("Response")
        const data = await response.json();
        removeLoadingMessage(loadingId);
        console.log("Remove Loading Message")
        if (!response.ok) {
            throw new Error(data.detail || `Error: ${response.status}`);
        }

        // Display bot response
        displayQueryResponse(data);
        console.log("Display query")
        // Save after bot responds
        saveSessionMessages(currentSessionId);
        console.log("Save Session Message")
        await fetch(`/sessions/${currentSessionId}/messages`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sessions.find(s => s.id === currentSessionId).messages)
        });
        console.log("Session API")
        // Update session name if needed
        updateSessionName(message);
        console.log("Update Session Name")
    } catch (error) {
        console.error('Error:', error);
        removeLoadingMessage(loadingId);
        addBotMessage(`Sorry, an error occurred: ${error.message}`);
        saveSessionMessages(currentSessionId);
    }
}

// Handle click on send button
sendButton.addEventListener('click', handleSendMessage);

// Handle Enter key inside textarea
userInput.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); // stop new line
        handleSendMessage();
    }
});


// Enter Key Press Event Handler inside Textarea
userInput.addEventListener('keydown', function(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault(); // Prevent adding a new line
        handleSendMessage();    // Trigger send
    }
});


// Update session name with first few words of first message
function updateSessionName(message) {
    const session = sessions.find(s => s.id === currentSessionId);
    if (!session) return;
    
    // Only update if it's using the default name format
    if (session.name.startsWith('Chat ')) {
        // Get first 3-4 words of message
        const words = message.split(' ');
        let newName = words.slice(0, Math.min(4, words.length)).join(' ');
        
        // Truncate if too long and add ellipsis
        if (newName.length > 30) {
            newName = newName.substring(0, 30) + '...';
        }
        
        // Update session name
        session.name = newName;
        
        // Update in storage
        localStorage.setItem('sessions', JSON.stringify(sessions));
        
        // Update UI
        loadSessions();
    }
}

// Check API health 
async function checkHealth() {
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            let message = 'API connection successful! ';
            if (data.ollama === 'connected') {
                message += 'Ollama service is connected.';
            } else {
                message += 'Warning: Ollama service is disconnected.';
            }
            addBotMessage(message);
        } else {
            addBotMessage('API health check failed.');
        }
    } catch (error) {
        console.error('Health check error:', error);
        addBotMessage('Failed to connect to the API. Please check if the server is running.');
    }
}

// Fetch available models
async function fetchAvailableModels() {
    try {
        const response = await fetch('/models');
        if (!response.ok) return;
        
        const data = await response.json();
        
        // Clear and repopulate model selector
        modelSelect.innerHTML = '';
        data.models.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = model;
            modelSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Error fetching models:', error);
    }
}

// Utility functions for displaying messages
function addUserMessage(content) {
    const escapedContent = escapeHtml(content);
    addUserMessageRaw(escapedContent);
}

function addUserMessageRaw(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message user-message';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = content;
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    scrollToBottom();
}

function addBotMessage(content) {
    addBotMessageRaw(content);
}

function addBotMessageRaw(content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message bot-message';
    
    const messageContent = document.createElement('div');
    messageContent.className = 'message-content';
    messageContent.innerHTML = content;
    
    messageDiv.appendChild(messageContent);
    chatMessages.appendChild(messageDiv);
    
    scrollToBottom();
}

function addLoadingMessage() {
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'loading';
    
    const spinner = document.createElement('div');
    spinner.className = 'loading-spinner';
    
    const loadingText = document.createElement('span');
    loadingText.textContent = 'Processing...';
    
    loadingDiv.appendChild(spinner);
    loadingDiv.appendChild(loadingText);
    chatMessages.appendChild(loadingDiv);
    
    scrollToBottom();
    return loadingId;
}

function removeLoadingMessage(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

function clearChat() {
    chatMessages.innerHTML = '';
    // addBotMessage('Chat cleared. How can I help you?');
}

function clearChatSilent() {
    chatMessages.innerHTML = '';
}

function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function autoResizeTextarea() {
    userInput.style.height = 'auto';
    userInput.style.height = (userInput.scrollHeight) + 'px';
    
    // Limit max height
    if (parseInt(userInput.style.height) > 150) {
        userInput.style.height = '150px';
    }
}