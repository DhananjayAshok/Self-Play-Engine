from self_play.chat_model import ChatModel


class InteractionChatModel(ChatModel):
    """A `ChatModel` backed by one seat's verifiers interaction.

    Each `complete` is one `interaction.turn`; with `StatelessHarness` the messages sent are
    the call's whole context. Once the run has ended (a limit, a timeout), further calls
    return an empty reply without touching the interaction, so the game's invalid-answer
    handling takes over.
    """

    def __init__(self, *, interaction):
        self._interaction = interaction
        self.terminated = False

    async def complete(self, *, messages: list[dict]) -> str:
        if self.terminated:
            return ""
        segment = await self._interaction.turn(messages)
        if segment.terminated:
            self.terminated = True
            return ""
        return segment.last_reply
