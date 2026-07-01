document.addEventListener('DOMContentLoaded', () => {
    // get all the sources (existing functionality)
    const sourceLinks = document.querySelectorAll('.sources-content a');

    sourceLinks.forEach(link => {
        link.setAttribute('target', '_blank');
        let currentRel = link.getAttribute('rel') || '';
        if (!currentRel.includes('noopener')) {
            link.setAttribute('rel', (currentRel + ' noopener noreferrer').trim());
        }
    });

    const chatWindow = document.getElementById('chat-window');
    if (chatWindow) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }

    // --- Utility Functions ---

    // Function to retrieve message content from the DOM
    const getMessageContent = (index) => {
        const messageElement = document.querySelector(`.chat-message[data-message-index="${index}"] .message-content`);
        return messageElement ? messageElement.innerText.trim() : 'Content Not Found';
    };



    // Function to send feedback to Flask
    const sendFeedback = (payload) => {
        fetch('/feedback', { // Use a dedicated feedback route
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
            // Optionally display a temporary "Thank you" message on success
        })
        .catch(error => {
            console.error('Error logging feedback:', error);
            // Optionally handle errors
        });
    };

    // --- Modal Elements ---
    const modal = document.getElementById('feedback-modal');
    const modalSubmitBtn = document.getElementById('modal-submit-btn');
    const modalCancelBtn = document.getElementById('modal-cancel-btn');
    const feedbackTextbox = document.getElementById('feedback-textbox');


    // --- Feedback Button Logic (Modified) ---
    const feedbackButtons = document.querySelectorAll('.js-feedback-btn');

    feedbackButtons.forEach(button => {
        button.addEventListener('click', (event) => {
            const clickedButton = event.currentTarget;
            const feedbackContainer = clickedButton.closest('.feedback-controls');

            const assistantIndex = feedbackContainer.dataset.assistantIndex;
            const userIndex = feedbackContainer.dataset.userIndex;
            const feedbackType = clickedButton.dataset.feedbackType; // 'up' or 'down'

            const allButtonsInContainer = feedbackContainer.querySelectorAll('.js-feedback-btn');
            const isCurrentlySelected = clickedButton.classList.contains('feedback-selected');

            // 1. Toggling Logic
            allButtonsInContainer.forEach(btn => {
                btn.classList.remove('feedback-selected');
            });
            let isSelected = false;

            if (!isCurrentlySelected) {
                clickedButton.classList.add('feedback-selected');
                isSelected = true;
            }

            // 2. Action based on feedback type and state
            if (feedbackType === 'up' && isSelected) {
                // Thumbs Up: Log immediately with no text feedback (as per original logic)
                const userContent = getMessageContent(userIndex);
                const assistantContent = getMessageContent(assistantIndex);


                const payload = {
                    thumbs_up_selected: true,
                    assistant_content: assistantContent,
                    user_content: userContent,
                    feedback_text: "no feedback", // 'no feedback' for thumbs up
                };
                sendFeedback(payload);

            } else if (feedbackType === 'down' && isSelected) {
                // Thumbs Down: Open modal to collect detailed feedback

                // Store message context in hidden modal fields
                document.getElementById('modal-user-content').value = getMessageContent(userIndex);
                document.getElementById('modal-assistant-content').value = getMessageContent(assistantIndex);
                document.getElementById('modal-assistant-index').value = assistantIndex;
                document.getElementById('modal-user-index').value = userIndex;

                // Reset the textbox and show the modal
                feedbackTextbox.value = '';
                modal.style.display = 'flex'; // Use 'flex' for easy centering

            } else {
                // Unselected (clicked on an already selected button) or Thumbs Up unselected:
                // If you want to log the removal of feedback, you'd add logic here.
                // For simplicity, we just stop.
            }

        });
    });

    // --- Modal Action Handlers ---

    // Cancel Button: Hides the modal and resets the UI state
    modalCancelBtn.addEventListener('click', () => {
        modal.style.display = 'none';
        // Remove the 'feedback-selected' class from ALL buttons
        const allButtons = document.querySelectorAll('.js-feedback-btn');
        allButtons.forEach(btn => btn.classList.remove('feedback-selected'));
    });

    // Submit Button: Collects text, sends AJAX, and hides modal
    modalSubmitBtn.addEventListener('click', () => {
        const detailedFeedback = feedbackTextbox.value.trim() || "No detailed text provided"; // Capture the text

        // Get context from the hidden fields
        const userContent = document.getElementById('modal-user-content').value;
        const assistantContent = document.getElementById('modal-assistant-content').value;
        const assistantIndex = document.getElementById('modal-assistant-index').value;

        // Find the 'thumbs down' button that corresponds to this message index
        const downButton = document.querySelector(`.feedback-controls[data-assistant-index="${assistantIndex}"] .js-feedback-btn[data-feedback-type="down"]`);

        // Check if the down button is currently selected (it should be)
        const isSelected = downButton && downButton.classList.contains('feedback-selected');


        if (isSelected) {
            // Send the payload with Thumbs Down selected and detailed text
            const payload = {
                thumbs_up_selected: false, // Thumbs down is clicked
                assistant_content: assistantContent,
                user_content: userContent,
                feedback_text: detailedFeedback, // Send the collected text
            };
            sendFeedback(payload);
        } else {
            // This shouldn't happen, but good for defensive coding
            console.warn("Feedback submitted, but 'Thumbs Down' wasn't selected on the UI.");
        }

        // Hide the modal after submission
        modal.style.display = 'none';
    });
});

// --- 🌟 INSERTED START: CSV Download Engine for Database Tables ---
    const csvDownloadButtons = document.querySelectorAll('.js-csv-download-btn');

    csvDownloadButtons.forEach(button => {
        button.addEventListener('click', (event) => {
            try {
                const rawJsonAttr = event.currentTarget.getAttribute('data-table-json');
                const tableDataArray = JSON.parse(rawJsonAttr);

                if (!tableDataArray || !tableDataArray.length) return;

                const columnHeaders = Object.keys(tableDataArray[0]);
                const csvDataLines = [columnHeaders.join(',')];

                for (const rowItem of tableDataArray) {
                    const rowCells = columnHeaders.map(header => {
                        let cellValue = rowItem[header] === null || rowItem[header] === undefined ? '' : String(rowItem[header]);

                        // Sanitize string content and escape quotes for standard CSV syntax compliance
                        cellValue = cellValue.replace(/"/g, '""');
                        if (cellValue.includes(',') || cellValue.includes('\n') || cellValue.includes('"')) {
                            cellValue = `"${cellValue}"`;
                        }
                        return cellValue;
                    });
                    csvDataLines.push(rowCells.join(','));
                }

                // Fire safe transient attachment download bridge frame right inside browser memory
                const blobStream = new Blob([csvDataLines.join('\n')], { type: 'text/csv;charset=utf-8;' });
                const virtualUrl = URL.createObjectURL(blobStream);

                const exportTrigger = document.createElement("a");
                exportTrigger.setAttribute("href", virtualUrl);
                exportTrigger.setAttribute("download", `Commercial_Intelligence_Export_${new Date().toISOString().split('T')[0]}.csv`);
                document.body.appendChild(exportTrigger);

                exportTrigger.click();
                document.body.removeChild(exportTrigger);
                URL.revokeObjectURL(virtualUrl);
            } catch (error) {
                console.error("Failed running browser file system export array transformation:", error);
            }
        });
    });
    // --- 🌟 INSERTED END ---
});