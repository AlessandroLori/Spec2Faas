
import re
from dataclasses import dataclass


@dataclass
class CodeBlock:
    code: str
    language: str


def extract_markdown_code_blocks(markdown_text: str) -> list[CodeBlock]:
    pattern = re.compile(r"```(?:\s*([\w\+\-]+))?\n([\s\S]*?)```")
    matches = pattern.findall(markdown_text)

    blocks = []

    for language, code in matches:
        blocks.append(
            CodeBlock(
                code=code,
                language=language.strip(),
            )
        )

    return blocks


def extract_markdown_code_string(markdown_text: str) -> str:
    return extract_markdown_code_blocks(markdown_text)[0].code
