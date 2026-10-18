"""학습·평가·추론에서 함께 쓰는 설정과 도우미 함수."""

KO2JE = "ko2je"  # 표준어 → 제주어
JE2KO = "je2ko"  # 제주어 → 표준어
DIRECTIONS = (KO2JE, JE2KO)

DIRECTION_TOKENS = {d: f"<{d}>" for d in DIRECTIONS}
DIRECTION_LABELS = {KO2JE: "표준어 → 제주어", JE2KO: "제주어 → 표준어"}

DEFAULT_MODEL = "gogamza/kobart-base-v2"
MAX_LENGTH = 128


def add_direction_tag(text: str, direction: str) -> str:
    """입력 문장 앞에 번역 방향 태그를 붙인다. 예: '<ko2je> 어서 오세요'"""
    if direction not in DIRECTION_TOKENS:
        raise ValueError(f"direction은 {DIRECTIONS} 중 하나여야 합니다: {direction!r}")
    return f"{DIRECTION_TOKENS[direction]} {text.strip()}"


def prepare_tokenizer(tokenizer):
    """방향 태그를 특수 토큰으로 등록하고 필수 특수 토큰을 보장한다.

    반환값: 새로 추가된 토큰 수 (0보다 크면 모델 임베딩 크기를 늘려야 함).
    """
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(DIRECTION_TOKENS.values())}
    )
    if tokenizer.pad_token is None:
        added += tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if tokenizer.eos_token is None:
        added += tokenizer.add_special_tokens({"eos_token": "</s>"})
    return added


def encode(tokenizer, text: str, max_length: int = MAX_LENGTH) -> list[int]:
    """KoBART 토크나이저는 EOS를 자동으로 붙이지 않으므로 직접 붙인다."""
    ids = tokenizer.encode(text, add_special_tokens=False)[: max_length - 1]
    return ids + [tokenizer.eos_token_id]
