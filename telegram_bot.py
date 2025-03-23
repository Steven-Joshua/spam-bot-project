import requests
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

# Telegram Bot Token
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

# Flask API URLs (Change according to your deployment)
FLASK_API_URL_TEXT = "https://spam-bot-project.onrender.com/test-text"
FLASK_API_URL_IMAGE = "https://spam-bot-project.onrender.com/test-image"

# Start Command
async def start(update: Update, context: CallbackContext):
    """Handles /start command."""
    await update.message.reply_text("🤖 Hello! Send me a message or an image, and I'll check if it's spam.")

# Function to Handle Text Messages
async def check_spam(update: Update, context: CallbackContext):
    """Handles text messages and checks for spam."""
    user_message = update.message.text
    chat_id = update.message.chat_id  # Using chat ID as 'source'

    payload = {"text": user_message, "source": "Telegram"}

    try:
        response = requests.post(FLASK_API_URL_TEXT, json=payload)
        response.raise_for_status()
        result = response.json()

        response_text = (
            f"📩 **Source:** {result.get('Source', 'Unknown')}\n"
            f"📝 **Text:** {result.get('Message Text', 'N/A')}\n"
            f"🔢 **Spam Probability:** {result.get('Spam Probability', 'N/A')}\n"
            f"⚠️ **Prediction:** {result.get('Prediction', 'N/A')}"
        )

    except requests.exceptions.RequestException as e:
        response_text = f"❌ Error processing the message.\n\n🔍 Debug Info:\n{e}"

    await update.message.reply_text(response_text)

# Function to Handle Image Messages
async def check_image(update: Update, context: CallbackContext):
    """Handles image uploads, extracts text, and checks for spam."""
    chat_id = update.message.chat_id
    photo = update.message.photo[-1]
    file = await photo.get_file()
    
    file_path = f"temp_{chat_id}.jpg"
    await file.download_to_drive(file_path)

    try:
        with open(file_path, "rb") as img_file:
            response = requests.post(
                FLASK_API_URL_IMAGE, 
                files={"file": img_file}, 
                data={"source": "Telegram"}
            )
            response.raise_for_status()
            result = response.json()

            response_text = (
                f"📩 **Source:** {result.get('Source', 'Unknown')}\n"
                f"📝 **Extracted Text:** {result.get('Extracted Text', 'N/A')}\n"
                f"🔢 **Spam Probability:** {result.get('Spam Probability', 'N/A')}\n"
                f"⚠️ **Prediction:** {result.get('Prediction', 'N/A')}"
            )

    except requests.exceptions.RequestException as e:
        response_text = f"❌ Error processing the image.\n\n🔍 Debug Info:\n{e}"

    finally:
        os.remove(file_path)

    await update.message.reply_text(response_text)

# Run the bot
def main():
    app = Application.builder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT, check_spam))
    app.add_handler(MessageHandler(filters.PHOTO, check_image))
    print("🤖 Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
