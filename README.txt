# SpamSniperBot - Telegram Spam Detection Bot

## Overview
SpamSniperBot is a Telegram bot that detects spam messages using a machine learning model.
Users can send a message, and the bot will classify it as spam or not.

## Features
- Uses an ML model to detect spam messages.
- Telegram bot interface for user interaction.
- Deployable via Flask & public hosting services.

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/SpamSniperBot.git
   cd SpamSniperBot
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. Start the bot:
   ```sh
   python app.py
   ```
2. Deploy the bot using Ngrok or a cloud service to get a public URL.

## Deployment
- Use **Ngrok** for local testing.
- Deploy on **Heroku, Railway, or Render** for a public URL.

## API Integration
- ML model runs on Flask (`app.py`) and processes messages via a POST request.

## License
This project is open-source under the MIT License.
