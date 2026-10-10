from verifiers.v1.harnesses.null.harness import NullHarness, NullHarnessConfig


class StatelessHarnessConfig(NullHarnessConfig):
    pass


class StatelessHarness(NullHarness):
    """The null chat harness, minus the growing conversation.

    verifiers sends every caller turn through `resume()`, whose default relaunches the chat
    program on the trace's previous messages plus the new ones. Here the new messages are
    the whole prompt: the engine rebuilds each call's full context itself, so each model
    call sees exactly what the player built and nothing accumulates.
    """

    async def resume(
        self,
        ctx,
        trace,
        runtime,
        endpoint,
        secret,
        mcp_urls,
        data,
        messages,
        tool_interception_url=None,
    ):
        kwargs = {"tool_interception_url": tool_interception_url} if self.SUPPORTS_TOOL_INTERCEPTION else {}
        return await self.launch(
            ctx,
            trace,
            runtime,
            endpoint,
            secret,
            mcp_urls,
            data.model_copy(update={"prompt": list(messages)}),
            **kwargs,
        )
