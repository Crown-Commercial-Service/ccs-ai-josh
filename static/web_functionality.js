document.addEventListener('DOMContentLoaded', () => {
    // 1. Format Source Links
    const sourceLinks = document.querySelectorAll('.sources-content a');
    sourceLinks.forEach(link => {
        link.setAttribute('target', '_blank');
        const currentRel = link.getAttribute('rel') || '';
        if (!currentRel.includes('noopener')) {
            link.setAttribute('rel', (currentRel + ' noopener noreferrer').trim());
        }
    });

    // 2. Element Selectors
    const loadingOverlay = document.getElementById('loading-overlay');
    const chatWindow = document.getElementById('chat-window');
    const chatForm = document.getElementById('chat-form');
    const chatMessageInput = document.getElementById('chat-message');
    const sendButton = chatForm ? chatForm.querySelector('button[type="submit"]') : null;

    if (chatWindow) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // 3. Loading Overlay Helper Functions
    const showLoading = () => {
        if (loadingOverlay) {
            loadingOverlay.hidden = false;
            loadingOverlay.style.display = 'flex';
        }
        if (chatMessageInput) {
            // CRITICAL: Use readOnly instead of disabled so Flask receives the form data
            chatMessageInput.readOnly = true;
        }
        // CRITICAL: Delay disabling the button by 1ms so the browser sends the POST request FIRST
        setTimeout(() => {
            if (sendButton) {
                sendButton.disabled = true;
            }
        }, 1);
    };

    const hideLoading = () => {
        if (loadingOverlay) {
            loadingOverlay.hidden = true;
            loadingOverlay.style.display = 'none';
        }
        if (chatMessageInput) {
            chatMessageInput.readOnly = false;
            chatMessageInput.disabled = false;
        }
        if (sendButton) {
            sendButton.disabled = false;
        }
    };

    // Always ensure the loader is hidden when the page reloads or navigates back
    window.addEventListener('pageshow', hideLoading);
    window.addEventListener('load', hideLoading);
    hideLoading();

    // 4. Form Submit Listener (Removed dual button click listener to prevent double triggers)
    if (chatForm && chatMessageInput) {
        chatForm.addEventListener('submit', () => {
            if (!chatMessageInput.checkValidity()) {
                return;
            }
            showLoading();
        });
    }

    // 5. Feedback System Logic
    const getMessageContent = (index) => {
        const messageElement = document.querySelector(`.chat-message[data-message-index="${index}"] .message-content`);
        return messageElement ? messageElement.innerText.trim() : 'Content Not Found';
    };

    const sendFeedback = (payload) => {
        fetch('/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            console.log('Feedback successfully logged:', data);
        })
        .catch(error => {
            console.error('Error logging feedback:', error);
        });
    };

    const modal = document.getElementById('feedback-modal');
    const modalSubmitBtn = document.getElementById('modal-submit-btn');
    const modalCancelBtn = document.getElementById('modal-cancel-btn');
    const feedbackTextbox = document.getElementById('feedback-textbox');
    const feedbackButtons = document.querySelectorAll('.js-feedback-btn');

    feedbackButtons.forEach(button => {
        button.addEventListener('click', (event) => {
            const clickedButton = event.currentTarget;
            const feedbackContainer = clickedButton.closest('.feedback-controls');

            const assistantIndex = feedbackContainer.dataset.assistantIndex;
            const userIndex = feedbackContainer.dataset.userIndex;
            const feedbackType = clickedButton.dataset.feedbackType;

            const allButtonsInContainer = feedbackContainer.querySelectorAll('.js-feedback-btn');
            const isCurrentlySelected = clickedButton.classList.contains('feedback-selected');

            allButtonsInContainer.forEach(btn => btn.classList.remove('feedback-selected'));
            let isSelected = false;

            if (!isCurrentlySelected) {
                clickedButton.classList.add('feedback-selected');
                isSelected = true;
            }

            if (feedbackType === 'up' && isSelected) {
                const userContent = getMessageContent(userIndex);
                const assistantContent = getMessageContent(assistantIndex);

                sendFeedback({
                    thumbs_up_selected: true,
                    assistant_content: assistantContent,
                    user_content: userContent,
                    feedback_text: 'no feedback',
                });
            } else if (feedbackType === 'down' && isSelected) {
                document.getElementById('modal-user-content').value = getMessageContent(userIndex);
                document.getElementById('modal-assistant-content').value = getMessageContent(assistantIndex);
                document.getElementById('modal-assistant-index').value = assistantIndex;
                document.getElementById('modal-user-index').value = userIndex;

                feedbackTextbox.value = '';
                if (modal) modal.style.display = 'flex';
            }
        });
    });

    if (modalCancelBtn) {
        modalCancelBtn.addEventListener('click', () => {
            if (modal) modal.style.display = 'none';
            const allButtons = document.querySelectorAll('.js-feedback-btn');
            allButtons.forEach(btn => btn.classList.remove('feedback-selected'));
        });
    }

    if (modalSubmitBtn) {
        modalSubmitBtn.addEventListener('click', () => {
            const detailedFeedback = feedbackTextbox.value.trim() || 'No detailed text provided';

            const userContent = document.getElementById('modal-user-content').value;
            const assistantContent = document.getElementById('modal-assistant-content').value;
            const assistantIndex = document.getElementById('modal-assistant-index').value;

            const downButton = document.querySelector(`.feedback-controls[data-assistant-index="${assistantIndex}"] .js-feedback-btn[data-feedback-type="down"]`);
            const isSelected = downButton && downButton.classList.contains('feedback-selected');

            if (isSelected) {
                sendFeedback({
                    thumbs_up_selected: false,
                    assistant_content: assistantContent,
                    user_content: userContent,
                    feedback_text: detailedFeedback,
                });
            } else {
                console.warn("Feedback submitted, but 'Thumbs Down' wasn't selected on the UI.");
            }

            if (modal) modal.style.display = 'none';
        });
    }
});