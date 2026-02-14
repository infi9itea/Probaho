$(document).ready(function () {
  let lineCounter = 1;
  let messageCount = 0;

  function addMessage(text, sender) {
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

  const RASA_URL = "http://localhost:8000/chat";

async function sendMessage() {
  const message = $('#user-input').val();
  if (!message.trim()) return;

  addMessage(message, 'user');
  $('#user-input').val('');

  try {
    const response = await fetch(RASA_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: 'user',
        message: message
      })
    });

    const data = await response.json();
    data.forEach(msg => addMessage(msg.text, 'bot'));

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