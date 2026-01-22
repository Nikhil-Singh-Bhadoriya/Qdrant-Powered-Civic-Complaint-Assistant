from __future__ import annotations
import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters

API_URL = os.environ.get("CIVICFIX_API_URL", "http://localhost:8000/v1/complaints/assist")
API_KEY = os.environ.get("CIVICFIX_API_KEY")
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send: city;ward_id;landmark;your issue text (photo optional).")

async def handle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import requests
    msg = update.message
    parts = [p.strip() for p in (msg.text or "").split(";", 3)]
    if len(parts) < 4:
        await msg.reply_text("Format: city;ward_id;landmark;issue text")
        return

    city, ward, landmark, issue = parts
    data = {"user_id": f"tg_{update.effective_user.id}", "city": city, "ward_id": ward, "landmark": landmark, "text": issue}
    headers = {"X-API-Key": API_KEY} if API_KEY else {}

    files = None
    if msg.photo:
        photo = msg.photo[-1]
        f = await photo.get_file()
        b = await f.download_as_bytearray()
        files = {"issue_photo": ("photo.jpg", bytes(b), "image/jpeg")}

    r = requests.post(API_URL, data=data, files=files, headers=headers)
    await msg.reply_text(r.text[:3500])

def main():
    if not TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN env var.")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.ALL, handle))
    app.run_polling()

if __name__ == "__main__":
    main()
