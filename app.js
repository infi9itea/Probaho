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

// 1. Update the URL to the Rasa REST webhook
const RASA_URL = "https://ftftr-ewu-chatbot-rasa.hf.space/webhooks/rest/webhook";

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
        sender: "user_123", // Rasa uses 'sender', not 'sender_id'
        message: message
      })
    });

    const data = await response.json(); // Rasa returns an array directly: [{text: "..."}]

    data.forEach(msg => {
      if (msg.text) {
        addMessage(msg.text, 'bot');
      }
    });

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