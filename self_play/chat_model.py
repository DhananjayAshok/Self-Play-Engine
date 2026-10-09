"""The one model interface the engine talks to.

Games and players only ever call `ChatModel.complete`. Real runs back it with a verifiers
interaction (`self_play_env.chat_model.InteractionChatModel`); unit tests use
`ScriptedChatModel`. This module must not import verifiers.
"""

from abc import ABC, abstractmethod

STOP_STRING = "[STOP]"
"""Models are told to end every reply with this. Seats also send it as a stop sequence."""


def strip_stop(*, text: str) -> str:
    """The reply up to the first `STOP_STRING` (all of it when absent), stripped."""
    return text.split(STOP_STRING)[0].strip()


class ChatModel(ABC):
    @abstractmethod
    async def complete(self, *, messages: list[dict]) -> str:
        """Reply to `messages`, which are the whole context for this call (the chat never
        grows between calls). Returns the raw reply text."""


class ScriptedChatModel(ChatModel):
    """Replies with the given strings in order; records every messages list it was sent."""

    def __init__(self, *, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[list[dict]] = []

    async def complete(self, *, messages: list[dict]) -> str:
        self.calls.append(messages)
        if not self._replies:
            raise RuntimeError("ScriptedChatModel ran out of replies")
        return self._replies.pop(0)
