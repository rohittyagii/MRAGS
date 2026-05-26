from __future__ import annotations

from mrags.models import Modality, RetrievedElement


class PromptBuilder:
    def __init__(self, system_prompt: str) -> None:
        """Time Complexity: O(1)
        Space Complexity: O(1)
        """
        self._system_prompt = system_prompt

    def build_messages(self, question: str, elements: list[RetrievedElement]) -> list[dict]:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        text_blocks: list[str] = []
        table_blocks: list[str] = []
        image_blocks: list[dict] = []
        for element in elements:
            if element.modality == Modality.TEXT:
                text_blocks.append(element.raw_content)
            elif element.modality == Modality.TABLE:
                table_blocks.append(element.raw_content)
            elif element.modality == Modality.IMAGE:
                image_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{element.raw_content}",
                        },
                    }
                )
        content: list[dict] = []
        if text_blocks:
            content.append({"type": "text", "text": _join_blocks("Text", text_blocks)})
        if table_blocks:
            content.append({"type": "text", "text": _join_blocks("Tables", table_blocks)})
        content.append({"type": "text", "text": f"Question: {question}"})
        content.extend(image_blocks)
        return [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": content},
        ]

    def build_text_prompt(self, question: str, elements: list[RetrievedElement]) -> str:
        """Time Complexity: O(N)
        Space Complexity: O(N)
        """
        sections: list[str] = [self._system_prompt]
        text_blocks: list[str] = []
        table_blocks: list[str] = []
        image_blocks: list[str] = []
        for element in elements:
            if element.modality == Modality.TEXT:
                text_blocks.append(element.raw_content)
            elif element.modality == Modality.TABLE:
                table_blocks.append(element.raw_content)
            elif element.modality == Modality.IMAGE:
                summary = element.embedded_summary or "Image summary unavailable."
                image_blocks.append(summary)
        if text_blocks:
            sections.append(_join_blocks("Text", text_blocks))
        if table_blocks:
            sections.append(_join_blocks("Tables", table_blocks))
        if image_blocks:
            sections.append(_join_blocks("Images", image_blocks))
        sections.append(f"Question:\n{question}")
        return "\n\n".join(sections)


def _join_blocks(title: str, blocks: list[str]) -> str:
    """Time Complexity: O(N)
    Space Complexity: O(N)
    """
    joined = "\n\n".join(blocks)
    return f"{title}:\n{joined}"
