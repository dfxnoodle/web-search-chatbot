class ChatBot {
    constructor() {
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.messagesContainer = document.getElementById('messages');
        this.status = document.getElementById('status');
        
        this.init();
    }

    init() {
        // Event listeners
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });

        // Focus on input
        this.messageInput.focus();
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;

        // Disable input and button
        this.setInputState(false);
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input
        this.messageInput.value = '';
        
        // Show typing indicator
        this.showTypingIndicator();
        
        try {
            const responseData = await this.callAPI(message);
            this.hideTypingIndicator();
            this.addStructuredMessage(responseData, 'bot');
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessage('Sorry, I encountered an error while processing your request. Please try again.', 'bot', true);
            console.error('Error:', error);
        } finally {
            this.setInputState(true);
            this.messageInput.focus();
        }
    }

    async callAPI(message) {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Network error');
        }

        const data = await response.json();
        return data; // Return full structured response data
    }

    addMessage(text, sender, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = sender === 'user' ? '👤' : '🤖';
        const avatarClass = sender === 'user' ? 'user-avatar' : 'bot-avatar';
        const textClass = isError ? 'text error-message' : 'text';
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="avatar ${avatarClass}">${avatar}</div>
                <div class="${textClass}">${this.formatMessage(text)}</div>
            </div>
        `;
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
   }

    addStructuredMessage(responseData, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const avatar = sender === 'user' ? '👤' : '🤖';
        const avatarClass = sender === 'user' ? 'user-avatar' : 'bot-avatar';
        
        // Create the main response content
        let content = `<div class="text">${this.formatMessage(responseData.response)}`;
        
        // Add sources if available
        if (responseData.sources && responseData.sources.length > 0) {
            content += '<div class="sources-section" style="margin-top: 15px; padding-top: 15px; border-top: 1px solid #e9ecef;">';
            content += '<div style="font-weight: 600; font-size: 0.9rem; color: #6c757d; margin-bottom: 10px;">📚 Sources:</div>';
            
            responseData.sources.forEach((source, index) => {
                if (source.url && source.url !== 'unknown') {
                    content += `
                        <div style="margin-bottom: 8px;">
                            <a href="${source.url}" target="_blank" rel="noopener noreferrer" 
                               style="color: #4facfe; text-decoration: none; font-size: 0.9rem;">
                                ${index + 1}. ${source.title || 'Web Source'}
                            </a>
                            ${source.snippet ? `<div style="font-size: 0.8rem; color: #6c757d; margin-top: 2px;">${source.snippet.substring(0, 100)}...</div>` : ''}
                        </div>
                    `;
                }
            });
            content += '</div>';
        }
        
        // Add metadata if available
        if (responseData.search_keywords || responseData.confidence) {
            content += '<div class="metadata-section" style="margin-top: 10px; font-size: 0.8rem; color: #6c757d;">';
            if (responseData.search_keywords) {
                content += `<div>🔍 Keywords: ${responseData.search_keywords}</div>`;
            }
            if (responseData.confidence) {
                const confidenceIcon = responseData.confidence === 'high' ? '🟢' : 
                                     responseData.confidence === 'medium' ? '🟡' : '🔴';
                content += `<div>📊 Confidence: ${confidenceIcon} ${responseData.confidence}</div>`;
            }
            content += '</div>';
        }
        
        content += '</div>';
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <div class="avatar ${avatarClass}">${avatar}</div>
                ${content}
            </div>
        `;
        
        this.messagesContainer.appendChild(messageDiv);
        this.scrollToBottom();
    }

    formatMessage(text) {
        // Basic text formatting
        return text
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>');
    }

    showTypingIndicator() {
        this.status.innerHTML = `
            <div class="typing-indicator">
                <span>Searching the web and analyzing content</span>
                <div class="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        `;
    }

    hideTypingIndicator() {
        this.status.innerHTML = '';
    }

    setInputState(enabled) {
        this.messageInput.disabled = !enabled;
        this.sendButton.disabled = !enabled;
        
        if (enabled) {
            this.status.textContent = '';
        }
    }

    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
}

// Health check function
async function checkServerHealth() {
    try {
        const response = await fetch('/api/health');
        if (response.ok) {
            console.log('Server is healthy');
        } else {
            console.warn('Server health check failed');
        }
    } catch (error) {
        console.error('Cannot connect to server:', error);
        document.getElementById('status').innerHTML = 
            '<span style="color: #dc3545;">⚠️ Cannot connect to server. Please make sure the backend is running.</span>';
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
    checkServerHealth();
});
