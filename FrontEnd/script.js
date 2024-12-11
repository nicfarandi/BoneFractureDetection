const form = document.querySelector(".chat-input");
const fileInput = document.getElementById("file");
const chatContent = document.querySelector(".chat-content");
const uploadButton = document.querySelector(".upload-button");
const chatBox = document.querySelector(".chat-box");
const chatBot = document.querySelector(".chat-bot");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const file = fileInput.files[0];
  console.log(file);
  if (!file) {
    return;
  }
  const uploadedImage = URL.createObjectURL(file);
  chatContent.innerHTML += `
    <div class="chat-user-message-container">
        <img
            class="chat-user-image"
            src="${uploadedImage}"
            alt=""
        />
    </div>
    <div class="chat-user-message-container">
        <div class="chat-user-message">
            <p class="chat-text">Please help me classify this image.</p>
        </div>
    </div>
  `;

  const loadingMessage = document.createElement("div");
  loadingMessage.classList.add("chat-bot-message-container");
  loadingMessage.innerHTML = `
    <img src="assets/chat-bot.png" class="chat-bot-icon" />
    <div class="chat-bot-message">
      <div class="loading-animation-container">
        <div class="loading-animation"></div>
        <div class="loading-animation"></div>
        <div class="loading-animation"></div>
      </div>
    </div>
  `;
  chatContent.appendChild(loadingMessage);

  chatBot.scrollTop = chatBot.scrollHeight;

  const formData = new FormData();
  formData.append("file", file);

  fileInput.value = "";

  try {
    const response = await fetch("http://127.0.0.1:5000/predict-image", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    const data = await response.json();

    chatContent.removeChild(loadingMessage);

    chatContent.innerHTML += `
      <div class="chat-bot-message-container">
        <img src="assets/chat-bot.png" class="chat-bot-icon" />
        <div class="chat-bot-message">
          <p class="chat-text">
            Classification complete! Our system detected the following:
            <span class="chat-text-bold">${data.predicted_class}</span>
          </p>
        </div>
      </div>
    `;

    console.log(data);
  } catch (error) {
    console.error(error);
    chatContent.removeChild(loadingMessage);
    chatContent.innerHTML += `
      <div class="chat-bot-message-container">
        <img src="assets/chat-bot.png" class="chat-bot-icon" />
        <div class="chat-bot-message">
          <p class="chat-text">
            We encountered an error while processing your request. Please try again later.
          </p>
        </div>
      </div>
    `;
  }
});
