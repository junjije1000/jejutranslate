"""학습된 모델로 번역하기.

명령줄 사용 예 (`pip install -e .` 후):
    jeju-translate --model outputs/final --direction ko2je "어서 오세요"
    jeju-translate --model outputs/final --direction je2ko "혼저 옵서예"
"""

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .common import DIRECTIONS, KO2JE, MAX_LENGTH, add_direction_tag, encode


class Translator:
    def __init__(self, model_path: str, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def translate(
        self,
        texts: str | list[str],
        direction: str = KO2JE,
        num_beams: int = 5,
        max_length: int = MAX_LENGTH,
        batch_size: int = 32,
    ):
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        results: list[str] = []
        pad_id = self.tokenizer.pad_token_id

        for i in range(0, len(texts), batch_size):
            batch = [encode(self.tokenizer, add_direction_tag(t, direction), max_length)
                     for t in texts[i : i + batch_size]]
            width = max(len(ids) for ids in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            attention = torch.zeros_like(input_ids)
            for row, ids in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                attention[row, : len(ids)] = 1

            out = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention.to(self.device),
                num_beams=num_beams,
                max_length=max_length,
            )
            results.extend(
                t.strip() for t in self.tokenizer.batch_decode(out, skip_special_tokens=True)
            )

        return results[0] if single else results


def main():
    p = argparse.ArgumentParser(description="표준어 ↔ 제주어 번역")
    p.add_argument("text", nargs="+", help="번역할 문장")
    p.add_argument("--model", default="outputs/final", help="모델 폴더 또는 Hugging Face Hub 이름")
    p.add_argument("--direction", choices=DIRECTIONS, default=KO2JE)
    p.add_argument("--num_beams", type=int, default=5)
    args = p.parse_args()

    tr = Translator(args.model)
    for src, tgt in zip(args.text, tr.translate(args.text, args.direction, args.num_beams)):
        print(f"{src}  →  {tgt}")


if __name__ == "__main__":
    main()
