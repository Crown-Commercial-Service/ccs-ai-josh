document.addEventListener('DOMContentLoaded', () => {
    // get all the sources
    const sourceLinks = document.querySelectorAll('.sources-content a');

    sourceLinks.forEach(link => {
        // Add target="_blank" which creates a new tab
        link.setAttribute('target', '_blank');

        // Add rel="noopener noreferrer" for security best practices when using _blank
        // This prevents the new page from accessing the old window object.
        let currentRel = link.getAttribute('rel') || '';
        if (!currentRel.includes('noopener')) {
            link.setAttribute('rel', (currentRel + ' noopener noreferrer').trim());
        }
    });

    // go back to the recent message
    const chatWindow = document.getElementById('chat-window');
    if (chatWindow) {
        chatWindow.scrollTop = chatWindow.scrollHeight;
    }
});