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

  fileInput.value = "";

  chatBot.scrollTop = chatBot.scrollHeight;
});
