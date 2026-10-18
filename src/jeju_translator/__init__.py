"""표준어 ↔ 제주어 양방향 번역기."""

from .common import DIRECTIONS, JE2KO, KO2JE, add_direction_tag

__all__ = ["DIRECTIONS", "KO2JE", "JE2KO", "add_direction_tag", "Translator"]


def __getattr__(name):
    # torch를 쓰지 않는 전처리 스크립트가 가볍게 import할 수 있도록 Translator는 필요할 때 불러온다.
    if name == "Translator":
        from .translate import Translator

        return Translator
    raise AttributeError(name)
