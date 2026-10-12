"""Episode construction helpers."""

from __future__ import annotations

from .types import GenerativeEpisode, GenerativeMode


def new_episode(
    *,
    episode_id: str,
    organism_id: str,
    root_state_id: str,
    mode: GenerativeMode,
    organism_step: int,
    generative_tick: int,
    target_id: str | None = None,
    source_episode_ids: tuple[str, ...] = (),
) -> GenerativeEpisode:
    return GenerativeEpisode(
        episode_id=episode_id,
        organism_id=organism_id,
        target_id=target_id,
        root_state_id=root_state_id,
        mode=mode,
        started_organism_step=organism_step,
        started_generative_tick=generative_tick,
        source_episode_ids=source_episode_ids,
    )


__all__ = ["new_episode"]
