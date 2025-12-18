from aiogram.filters import CommandStart, Command
from aiogram.types import Message

from bot.keyboards import build_main_reply_kb, build_settings_kb, build_history_kb
from bot.dispatcher import router, logger
from storage.state import pop_searches, pop_awaiting, get_history


@router.message(CommandStart())
async def cmd_start(msg: Message) -> None:
    """Команда /start — сбрасывает состояние и показывает инструкцию.

    Args:
        msg (Message): Входящее сообщение.
    """
    uid = msg.from_user.id if msg.from_user is not None else None
    if uid is not None:
        pop_searches(uid)
        pop_awaiting(uid)
    logger.info("Команда /start от пользователя %s", str(uid))
    await msg.answer(
        "✨ Отправьте ссылку — скачаю по вашим настройкам.\n"
        "📝 Или отправьте название — покажу список из 25 результатов.\n"
        "🍪 Если нужен доступ — пришлите файл cookies.txt.",
        reply_markup=build_main_reply_kb(),
    )


@router.message(Command("menu"))
async def cmd_menu(msg: Message) -> None:
    await cmd_start(msg)


@router.message(Command("help"))
async def cmd_help(msg: Message) -> None:
    """Команда /help — краткая справка.

    Args:
        msg (Message): Сообщение команды.
    """
    logger.info(
        "Команда /help от пользователя %s",
        str(msg.from_user.id if msg.from_user else None),
    )
    text = (
        "ℹ️ Что умеет бот:\n"
        "1. 🔗 Отправьте ссылку — скачивание пойдёт в выбранном режиме.\n"
        "2. 🔎 Напишите запрос — получаете до 25 результатов.\n"
        "3. ⚙️ Команда /settings или кнопка меню меняет режим по умолчанию.\n"
        "\n"
        "🍪 Cookies (cookies.txt):\n"
        "• Это экспорт авторизационных данных сайта (YouTube и др.).\n"
        "• Нужны, если видео приватное, доступно только после входа или выдаёт защиту.\n"
        "• Получите через расширение браузера Get cookies.txt / EditThisCookie и сохраните файл cookies.txt.\n"
        "• Файл живёт недолго: чаще всего до закрытия сайта.\n"
        "• Максимальный размер 5МБ — отправляйте как документ.\n"
        "\n"
        "После загрузки cookies бот автоматически повторит поиск или скачивание."
    )
    await msg.answer(text, reply_markup=build_main_reply_kb())


@router.message(Command("settings"))
async def cmd_settings(msg: Message) -> None:
    """Открывает меню настроек.

    Args:
        msg (Message): Сообщение команды.
    """
    if msg.from_user is None:
        await msg.answer(
            "⚙️ Настройки недоступны для этого типа сообщения.",
            reply_markup=build_main_reply_kb(),
        )
        return
    logger.info("Открытие настроек пользователем %s", str(msg.from_user.id))
    await msg.answer(
        "⚙️ Настройки типа скачивания:",
        reply_markup=build_settings_kb(msg.from_user.id).as_markup(),
    )

@router.message(Command("history"))
async def cmd_history(msg: Message) -> None:
    """Открывает историю загрузок пользователя (inline keyboard).

    Args:
        msg (Message): Сообщение команды.
    """
    if msg.from_user is None:
        await msg.answer(
            "📜 История недоступна для этого типа сообщения.",
            reply_markup=build_main_reply_kb(),
        )
        return
    uid = msg.from_user.id
    logger.info("Открытие истории пользователем %s", str(uid))
    items = get_history(uid)
    if not items:
        await msg.answer("ℹ️ История пуста.", reply_markup=build_main_reply_kb())
        return
    await msg.answer("📜 Ваша история загрузок:", reply_markup=build_history_kb(uid).as_markup())