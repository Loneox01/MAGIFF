"""Compact proactive Discord messages for automatic lineup reviews."""

from __future__ import annotations

from .discord import DISCORD_MESSAGE_LIMIT, DiscordBotClient
from lineups.reviews import AutomaticLineupReview, ReviewOutcome, ReviewTrigger


PING_OUTCOMES = {
    ReviewOutcome.CHANGE_RECOMMENDED,
    ReviewOutcome.REVIEW_FAILED,
    ReviewOutcome.EMERGENCY_UPDATE,
}


class DiscordLineupNotifier:
    def __init__(self, bot_token: str, channel_id: str, *, client=None) -> None:
        self.channel_id = channel_id
        self.bot = DiscordBotClient(bot_token, client=client)

    def send(self, content: str) -> str:
        return self.bot.send_channel_message(
            channel_id=self.channel_id,
            content=content,
        )


def _player_line(player) -> str:
    projection = (
        f"{player.projected_points:.1f} proj"
        if player.projected_points is not None
        else "no projection"
    )
    designation = f", {player.injury_code}" if player.injury_code else ""
    slot = player.current_slot_id or "bench"
    return (
        f"- **{player.display_name}** ({player.team or '-'} {player.position}, "
        f"{slot}, {projection}{designation})"
    )


def _change_line(change) -> str:
    return (
        f"- **{change.slot_id}:** {change.outgoing_player or 'empty'} -> "
        f"{change.incoming_player or 'empty'}"
    )


def format_lineup_review(
    review: AutomaticLineupReview,
    *,
    owner_discord_user_id: str,
) -> str:
    """Render exactly one bounded message for one kickoff slate."""
    should_ping = review.outcome in PING_OUTCOMES
    mention = f"<@{owner_discord_user_id}>\n" if should_ping else ""
    title = {
        ReviewOutcome.NO_CHANGE: "NO CHANGE",
        ReviewOutcome.CHANGE_RECOMMENDED: "CHANGE RECOMMENDED",
        ReviewOutcome.REVIEW_FAILED: "REVIEW FAILED",
        ReviewOutcome.EMERGENCY_UPDATE: "EMERGENCY UPDATE",
    }[review.outcome]
    if review.trigger == ReviewTrigger.E2E:
        title = f"E2E TEST - {title}"

    lines = [mention + f"## MAGIFF Lineup Review - {title}"]
    if review.slate is not None:
        timestamp = int(review.slate.kickoff_at.timestamp())
        lines.append(
            f"**Week {review.context.week} slate:** <t:{timestamp}:F> "
            f"(<t:{timestamp}:R>)"
        )
        if review.slate.starters:
            lines.extend(["**Starters locking**", *map(_player_line, review.slate.starters)])
        if review.slate.bench:
            lines.extend(["**Bench locking**", *map(_player_line, review.slate.bench)])

    if review.error:
        lines.extend(["**Problem**", review.error])
    elif review.immediate_changes:
        lines.extend(
            ["**Make before this slate locks**", *map(_change_line, review.immediate_changes)]
        )
    else:
        lines.append("**Decision:** Keep the current lineup through this deadline.")

    if review.provisional_changes:
        lines.extend(
            ["**Provisional later changes**", *map(_change_line, review.provisional_changes[:3])]
        )
    if review.agent_result is not None:
        strategy = review.agent_result.analysis.overall_strategy.strip()
        if strategy:
            lines.extend(["**Why**", strategy])
        warnings = review.agent_result.analysis.warnings[:2]
        if warnings:
            lines.extend(["**Recheck**", *[f"- {warning}" for warning in warnings]])

    message = "\n".join(lines).strip()
    if len(message) > DISCORD_MESSAGE_LIMIT:
        marker = "\n... details truncated; full review is stored in Supabase."
        message = message[: DISCORD_MESSAGE_LIMIT - len(marker)].rstrip() + marker
    return message
