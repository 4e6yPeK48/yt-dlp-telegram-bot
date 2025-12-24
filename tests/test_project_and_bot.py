# python
import os
import tempfile
import shutil
import math

import pytest

from config import MAX_QUERY_LEN, PAGE_SIZE
from utils.text import (
    sanitize_query,
    make_caption,
    make_multiline_caption,
    format_duration_hms,
    parse_main_button_intent,
)
from utils.validators import is_url, is_youtube_url, is_audio_platform
from services.media import norm_base, extract_id_from_base
from services.ytdlp import decide_effective_mode, make_duration_match_filter
from storage.state import (
    StateStore,
    set_searches,
    pop_searches,
    set_user_mode,
    add_download_history,
    clear_history,
    reset_history_page,
)
from bot.keyboards import (
    build_results_kb,
    build_settings_kb,
    build_download_choice_kb,
    build_history_kb,
)
from aiogram.types import InlineKeyboardMarkup


class TestProjectCore:
    def test_sanitize_and_captions_and_duration(self):
        # sanitize_query удаляет управляющие символы и усекает строку
        raw = "hello\x00\x1f world" + ("x" * (MAX_QUERY_LEN + 10))
        s = sanitize_query(raw)
        assert "\x00" not in s and "\x1f" not in s
        assert len(s) <= MAX_QUERY_LEN

        # make_caption обрезает пробелы и усекает по лимиту
        long = "A" * (1002)
        cap = make_caption(long, limit=50)
        assert len(cap) <= 50

        # мультистрочный заголовок сохраняет переносы строк и убирает управляющие символы
        multi = "Line1\r\nLine2\x00\nLine3"
        mm = make_multiline_caption(multi, limit=200)
        assert "Line1" in mm and "Line2" in mm and "\x00" not in mm

        # форматирование длительности
        assert format_duration_hms(45) == "00:45"
        assert format_duration_hms(3605) == "01:00:05"
        assert format_duration_hms(None) == "—"

    def test_validators_basic(self):
        assert is_url("https://example.com")
        assert not is_url("notaurl")
        assert is_youtube_url("https://youtube.com/watch?v=abc")
        assert is_youtube_url("https://youtu.be/abc")
        assert not is_youtube_url("https://example.com/video")
        assert is_audio_platform("https://music.youtube.com/some")
        assert is_audio_platform("https://soundcloud.com/artist/track")
        assert not is_audio_platform("https://example.com/")

    def test_state_store_pending_and_awaiting_and_pages(self, tmp_path):
        ss = StateStore(cookies_dir=str(tmp_path))
        # токен для ожидаемого URL
        t1 = ss.save_pending_url(10, "https://u.test/1")
        got = ss.get_pending(t1)
        assert got and got["user_id"] == 10
        popped = ss.pop_pending(t1)
        assert popped and popped["url"] == "https://u.test/1"
        assert ss.get_pending(t1) is None

        # ожидание запроса cookies
        ss.remember_cookie_request(
            11, kind="download", url="https://u.test/2", mode="audio"
        )
        aw = ss.get_awaiting(11)
        assert aw and aw["kind"] == "download" and aw["url"].endswith("/2")
        popped_aw = ss.pop_awaiting(11)
        assert popped_aw and popped_aw.get("asked") is True
        assert ss.get_awaiting(11) is None

        # путь к cookies располагается под tmp_path
        cookies_path = ss.get_user_cookies_path(12)
        assert cookies_path.startswith(str(tmp_path))

        # поведение slice_page (постраничная нарезка)
        items = list(range(10))
        page0, pages = ss.slice_page(items, 0, page_size=3)
        assert pages == math.ceil(len(items) / 3)
        assert page0 == [0, 1, 2]
        page2, _ = ss.slice_page(items, 2, page_size=3)
        assert page2 == [6, 7, 8]

    def test_make_duration_match_filter_and_decide_mode_and_norms(self):
        f = make_duration_match_filter(30)
        # инфо с короткой длительностью => None (разрешено)
        assert f({"duration": 10}) is None
        # длинная => строка с причиной
        assert isinstance(f({"duration": 3600}), str) and "duration" in f(
            {"duration": 3600}
        )

        # decide_effective_mode
        assert (
            decide_effective_mode("auto", "https://music.youtube.com/track") == "audio"
        )
        assert decide_effective_mode("auto", "https://youtube.com/watch?v=1") == "video"
        assert (
            decide_effective_mode("audio", "https://youtube.com/watch?v=1") == "audio"
        )

        # помощники для медиа
        assert norm_base("/path/to/File Name [abcd123].mp3") == "File Name [abcd123]"
        assert extract_id_from_base("Name [ABCdef12] more") == "ABCdef12"
        assert extract_id_from_base("noidhere") is None


class TestBotComponents:
    def setup_method(self):
        # убедиться, что разделяемое singleton-хранилище очищено перед тестами клавиатур
        try:
            pop_searches(9999)
        except Exception:
            pass
        try:
            clear_history(9999)
            reset_history_page(9999)
        except Exception:
            pass

    def teardown_method(self):
        try:
            pop_searches(9999)
        except Exception:
            pass
        try:
            clear_history(9999)
            reset_history_page(9999)
        except Exception:
            pass

    def test_build_results_kb_and_navigation(self):
        # заполняем результаты для пользователя 9999
        results = [
            {"title": f"Title {i}", "url": f"https://u/{i}", "duration": i * 10}
            for i in range(7)
        ]
        set_searches(9999, {"results": results, "page": 0})
        kb = build_results_kb(9999).as_markup()
        assert isinstance(kb, InlineKeyboardMarkup)
        # текст первой кнопки содержит Title 0
        assert kb.inline_keyboard[0][0].text.startswith("Title 0")
        # строка навигации присутствует (последняя строка перед отменой)
        last_row = kb.inline_keyboard[
            -2
        ]  # вторая с конца строка — навигация, когда есть результаты
        assert any("Назад" in b.text or "Вперёд" in b.text for b in last_row)
        pop_searches(9999)

    def test_build_settings_and_download_choice_and_history_kb(self):
        # настройки: ставим режим audio и проверяем чекбокс
        set_user_mode(8888, "audio")
        sk = build_settings_kb(8888).as_markup()
        assert any(
            btn.text.startswith("✅") and "Только аудио" in btn.text
            for row in sk.inline_keyboard
            for btn in row
        )

        # клавиатура выбора скачивания содержит токен в callback_data
        token = "tokentest123"
        dk = build_download_choice_kb(8888, token).as_markup()
        found = False
        for row in dk.inline_keyboard:
            for btn in row:
                if btn.callback_data and f"dl:audio:{token}" in btn.callback_data:
                    found = True
        assert found

        # клавиатура истории: добавляем историю и проверяем создание кнопки
        add_download_history(
            7777,
            {
                "url": "https://h/1",
                "mode": "audio",
                "title": "HistTitle",
                "duration": 42,
            },
        )
        hk = build_history_kb(7777).as_markup()
        assert isinstance(hk, InlineKeyboardMarkup)
        # первая кнопка данных должна ссылаться на history:show:0
        found_show = any(
            btn.callback_data and btn.callback_data.startswith("history:show:")
            for row in hk.inline_keyboard
            for btn in row
        )
        assert found_show
        clear_history(7777)
        reset_history_page(7777)
