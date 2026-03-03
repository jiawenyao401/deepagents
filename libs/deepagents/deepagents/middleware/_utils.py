"""中间件公共工具函数。"""

from langchain_core.messages import ContentBlock, SystemMessage


def append_to_system_message(
    system_message: SystemMessage | None,
    text: str,
) -> SystemMessage:
    """向系统消息追加文本内容。

    若原系统消息已有内容，则在追加文本前自动插入两个换行符以保持格式整洁。
    若原系统消息为 None，则创建一个仅包含新文本的系统消息。

    Args:
        system_message: 已有的系统消息，或 None（表示尚无系统消息）。
        text: 需要追加到系统消息末尾的文本。

    Returns:
        追加了新文本后的新 SystemMessage 对象。
    """
    # 若已有系统消息，则复制其内容块列表；否则从空列表开始
    new_content: list[ContentBlock] = list(system_message.content_blocks) if system_message else []

    # 若已有内容，则在新文本前加两个换行符，保证段落间距
    if new_content:
        text = f"\n\n{text}"

    # 将新文本作为 text 类型的内容块追加
    new_content.append({"type": "text", "text": text})

    return SystemMessage(content_blocks=new_content)
