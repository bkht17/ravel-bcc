"""Оценка качества распознанного (OCR) и подозрительного текста: словарь частых слов и смесь алфавитов."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Порог доли токенов из словаря частых слов. Подобран по фикстурам: samples/requirements.md и PDF
# из него дают 0.23 (таблицы и идентификаторы разбавляют текст), связный английский — 0.7,
# tests/fixtures/scan.pdf, распознанный английской моделью («Het OGwme ceegenna»), — 0.04.
# 0.12 — примерно середина между 0.04 и 0.23.
MIN_KNOWN_SHARE = 0.12

# Порог доли слов со смесью кириллицы и латиницы внутри одного слова. В нормальном тексте таких
# слов почти нет (0–2 %: опечатки, «Ф-1» режется дефисом); при OCR чужим алфавитом и при битых
# кодировках шрифтов доля заметно больше. 0.10 — запас над нормой в 5 раз.
MAX_MIXED_SHARE = 0.10

# Для текстового слоя (не OCR) словарь не применяем (документ может быть на казахском или другом
# языке), смотрим только на смесь алфавитов; порог выше, чтобы не ловить редкие вставки.
MAX_MIXED_SHARE_NATIVE = 0.20

# Меньше стольких слов судить о качестве нельзя: для OCR это признак провала распознавания.
MIN_TOKENS = 8

_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
_CYR = re.compile(r"[А-Яа-яЁё]")
_LAT = re.compile(r"[A-Za-z]")

_RU = (
    "и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было "
    "вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас "
    "нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их "
    "чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой "
    "совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при "
    "наконец два об другой хоть после над больше тот через эти нас про них какая много разве три эту "
    "моя впрочем свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между "
    "должен должна должно должны система пользователь клиент требование данные документ при этом "
    "которые который которая которое также либо каждый каждого может быть будут являются"
).split()

_EN = (
    "the of and to in a is that for it as was with be by on not he this are or his from at which but "
    "have an had they you were their one all we can her has there been if more when will would who so "
    "no system user shall should must may data when each any other than then into only its also these "
    "use used using set value time such after before must be able"
).split()

KNOWN_WORDS = frozenset(_RU) | frozenset(_EN)


@dataclass(frozen=True)
class TextQuality:
    tokens: int
    known_share: float
    mixed_share: float


def assess(text: str) -> TextQuality:
    tokens = _WORD.findall(text)
    # «push-уведомление» режется на «push» и «уведомление»: дефис не смешивает алфавиты
    n = len(tokens)
    if n == 0:
        return TextQuality(0, 0.0, 0.0)
    known = sum(1 for t in tokens if t.lower().replace("ё", "е") in KNOWN_WORDS)
    mixed = sum(1 for t in tokens if _CYR.search(t) and _LAT.search(t))
    return TextQuality(n, known / n, mixed / n)


def ocr_is_poor(q: TextQuality) -> bool:
    return q.tokens < MIN_TOKENS or q.known_share < MIN_KNOWN_SHARE or q.mixed_share > MAX_MIXED_SHARE


def native_is_suspicious(q: TextQuality) -> bool:
    return q.tokens >= MIN_TOKENS and q.mixed_share > MAX_MIXED_SHARE_NATIVE
