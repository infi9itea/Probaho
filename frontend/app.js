$(document).ready(function() {

    function addMessage(text, sender) {
        const msgDiv = $('<div>').addClass('message ' + sender).text(text);
        $('#messages').append(msgDiv);
        $('#chat-window').scrollTop($('#chat-window')[0].scrollHeight);
    }

    async function sendMessage() {
        const message = $('#user-input').val();
        if (!message.trim()) return;

        addMessage(message, 'user');
        $('#user-input').val('');

        try {
            const response = await fetch('http://localhost:8000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: 'test_user', message: message })
            });
            const data = await response.json();

            data.forEach(msg => {
                addMessage(msg.text, 'bot');
            });

        } catch (err) {
            addMessage('Error connecting to server.', 'bot');
            console.error(err);
        }
    }

    $('#send-btn').click(sendMessage);
    $('#user-input').keypress(function(e) {
        if (e.which === 13) sendMessage();
    });

});
