import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext

TOKEN = ""
FLASK_API_URL_TEXT =  "" # Adjust if Flask is hosted elsewhere
FLASK_API_URL_IMAGE = "" # Adjust if Flask is hosted elsewhere

# Start Command
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Hello! Send me a message or an image, and I'll check if it's spam.")

# Function to Handle Text Messages
async def check_spam(update: Update, context: CallbackContext):
    user_message = update.message.text
    chat_id = update.message.chat_id  # Use chat ID as 'source'

    payload = {"text": user_message, "source": str(chat_id)}
    response = requests.post(FLASK_API_URL_TEXT, json=payload)

    if response.status_code == 200:
        result = response.json()
        response_text = f"📩 **Source:** {result['Source']}\n" \
                        f"📝 **Text:** {result['Message Text']}\n" \
                        f"🔢 **Spam Probability:** {result['Spam Probability']}\n" \
                        f"⚠️ **Prediction:** {result['Prediction']}"
    else:
        response_text = "❌ Error processing the message."

    await update.message.reply_text(response_text)

# Function to Handle Images
async def check_image(update: Update, context: CallbackContext):
    chat_id = update.message.chat_id  # Use chat ID as 'source'
    file = await update.message.photo[-1].get_file()
    file_path = await file.download()

    with open(file_path, "rb") as img_file:
        response = requests.post(FLASK_API_URL_IMAGE, files={"file": img_file}, data={"source": str(chat_id)})

    if response.status_code == 200:
        result = response.json()
        response_text = f"📩 **Source:** {result['Source']}\n" \
                        f"📝 **Extracted Text:** {result['Extracted Text']}\n" \
                        f"🔢 **Spam Probability:** {result['Spam Probability']}\n" \
                        f"⚠️ **Prediction:** {result['Prediction']}"
    else:
        response_text = "❌ Error processing the image."

    await update.message.reply_text(response_text)

# Main Function to Set Up Bot
def main():
    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, check_spam))
    app.add_handler(MessageHandler(filters.PHOTO, check_image))  # Handle image uploads

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
