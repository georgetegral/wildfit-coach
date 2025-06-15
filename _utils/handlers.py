from telegram import Update
from telegram.ext import ContextTypes

# We'll get the RAG service from the main app context instead of creating a new instance
# This will be passed or accessed through a global reference

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    await update.message.reply_text('¡Hola! Soy Jorge, tu coach de WildFit. Puedo responder tus preguntas sobre nutrición, estaciones metabólicas y el programa en general. También puedo darte apoyo emocional y motivarte a seguir el programa. ¡Estoy aquí para ayudarte!')


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a help message when the /help command is issued."""
    await update.message.reply_text('Simplemente escribe tu pregunta e intentaré responderla basándome en los principios de WildFit.')


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles regular text messages and queries the RAG service."""
    if not update.message or not update.message.text:
        return

    user_question = update.message.text
    chat_id = update.message.chat_id

    # Import here to avoid circular imports and use the global instance
    from _utils.rag_service import RAGService
    
    # Create a new instance or get the global one
    # In a production environment, you'd want to share the instance
    rag_service = RAGService()
    if not rag_service.load_index():
        await update.message.reply_text("Lo siento, aún no estoy listo para responder preguntas. Por favor, inténtalo de nuevo en un momento.")
        return

    try:
        await context.bot.send_chat_action(chat_id=chat_id, action="typing")
        result = rag_service.query(user_question)
        
        answer = result.get("answer", "Lo siento, no pude encontrar una respuesta para eso.")
        sources = result.get("sources", [])
        print(f"🤔 Pregunta: {user_question}")
        print(f"🤖 Respuesta generada: {answer}")
        print(f"📚 Fuentes utilizadas: {sources}")
        await update.message.reply_text(answer)

    except Exception as e:
        print(f"❌ Error processing Telegram message: {e}")
        await update.message.reply_text("Encontré un error al intentar responder tu pregunta. Por favor, inténtalo de nuevo más tarde.")