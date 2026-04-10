$(document).ready(function () {
  let lineCounter = 1;
  let messageCount = 0;

  // Generate or retrieve session_id
  let session_id = localStorage.getItem('session_id');
  if (!session_id) {
    session_id = 'session_' + Math.random().toString(36).substr(2, 9);
    localStorage.setItem('session_id', session_id);
  }

  function addMessage(text, sender) {
    if (!text) return;
    const lines = text.split('\n').length;

    const msgDiv = $('<div>')
      .addClass('message ' + sender)
      .attr('data-line', lineCounter)
      .append($('<span>').text(text));

    $('#messages').append(msgDiv);
    $('#chat-window').scrollTop($('#chat-window')[0].scrollHeight);

    lineCounter += lines;
    messageCount++;

    $('#line-number').text(lineCounter);
    $('#status-right').text(messageCount + ' messages');
  }

  const CHAT_URL = "/chat";

  async function sendMessage() {
    const message = $('#user-input').val();
    if (!message.trim()) return;

    addMessage(message, 'user');
    $('#user-input').val('');

    try {
      const response = await fetch(CHAT_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          session_id: session_id,
          message: message
        })
      });

      const data = await response.json();
      if (Array.isArray(data)) {
        data.forEach(msg => {
          if (msg.text) addMessage(msg.text, 'bot');
        });
      } else if (data.error) {
        addMessage('Error: ' + data.error, 'bot');
      }

    } catch (err) {
      addMessage('Error connecting to server.', 'bot');
      console.error(err);
    }
  }

  $('#user-input').on('keydown', function (e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
});
