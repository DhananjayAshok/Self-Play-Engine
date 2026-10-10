from collections.abc import Iterator
from itertools import count

import verifiers.v1 as vf


class SelfPlayData(vf.TaskData):
    info: dict
    """The game's RNG seed; the game is reproducible from it."""


class SelfPlayTask(vf.Task[SelfPlayData]):
    pass


class SelfPlayTasksetConfig(vf.TasksetConfig):
    pass


class SelfPlayTaskset(vf.Taskset[SelfPlayTask, SelfPlayTasksetConfig]):
    INFINITE = True

    def load(self) -> Iterator[SelfPlayTask]:
        for i in count():
            yield SelfPlayTask(SelfPlayData(name=f"game#{i}", prompt=None, info={"seed": i}))
