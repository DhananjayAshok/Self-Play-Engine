"""Every game, by the name a run config uses (`[env] game = "..."`)."""

from self_play.game import Game
from self_play.games.copycat_rps import CopycatRPSGame
from self_play.games.coup import CoupGame
from self_play.games.sheriff import SheriffGame

GAMES: dict[str, type[Game]] = {game.name: game for game in (CopycatRPSGame, CoupGame, SheriffGame)}
