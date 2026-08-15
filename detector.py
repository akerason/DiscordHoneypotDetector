import asyncio
from collections import Counter
from typing import Any, AsyncGenerator


# discord.py-self
import discord
import discord.ext.commands

import joblib

# Load the trained classifier
embedder, clf = joblib.load('detector_model.pkl')

def prediction(x):
    """
    0.0 - 0.2 -> Low chance
    0.2 - 0.4 -> Mid chance
    0.4 - 0.6 -> High chance
    0.6 - 1.0 -> Almost certain

    This return a number between 0 and 1, so that you can use a heuristic, based on score, to avoid false positives
    """
    emb = embedder.encode([x])
    prob = clf.predict_proba(emb)[0][1]
    return float(prob)

async def async_prediction(x):
    """Async version of prediction"""
    return await asyncio.to_thread(prediction,x)

async def find_safe_channels(bot: discord.ext.commands.Bot, server: discord.Guild, max_score=2.3, minimal_score=0.2, debug=False):
    member = server.get_member(bot.user.id)
    if not member:
        return

    for channel in server.channels:
        if not isinstance(channel, discord.TextChannel):
            continue

        perms = channel.permissions_for(member)
        if not perms.send_messages or not perms.read_message_history:
            continue

        score = 0.0

        def add_score(amount, reason):
            nonlocal score
            old_score = score
            if amount < minimal_score:
                amount = 0.0
            score += amount
            if debug:
                print(
                    f"[{channel.name}] "
                    f"{reason}: +{amount:.4f} "
                    f"({old_score:.4f} -> {score:.4f})"
                )

        if debug:
            print(f"\n=== Checking #{channel.name} ===")

        if channel.name:
            value = await async_prediction(channel.name)
            add_score(value, f"channel name {channel.name!r}")

        if channel.topic:
            value = await async_prediction(channel.topic)
            add_score(value, f"channel topic {channel.topic!r}")

        messages = [message async for message in channel.history(limit=10)]

        if debug:
            print(f"[{channel.name}] Found {len(messages)} messages")

        if len(messages) < 4:
            add_score(0.2, f"only {len(messages)} messages")

        authors = []

        for message in messages:
            authors.append(message.author.id)

            if len(message.reactions) > 5:
                for reaction in message.reactions:
                    if reaction.count > 10:
                        add_score(
                            0.2,
                            f"message {message.id}: reaction count={reaction.count}"
                        )

            if score >= max_score:
                break

            if message.embeds:
                embed = message.embeds[0]

                if embed.title:
                    value = await async_prediction(embed.title)
                    add_score(
                        value,
                        f"message {message.id}: embed title {embed.title!r}"
                    )

                if embed.description:
                    value = await async_prediction(embed.description)
                    add_score(
                        value,
                        f"message {message.id}: embed description"
                    )

            if message.content:
                value = await async_prediction(message.content)
                add_score(
                    value,
                    f"message {message.id}: content {message.content!r}"
                )

            if score >= max_score:
                break

        if authors:
            author_count = Counter(authors).most_common(1)[0][1]

            if author_count > len(authors) * 0.75:
                add_score(
                    0.4,
                    f"one author wrote {author_count}/{len(authors)} messages"
                )

        print(f"[{channel.name}] FINAL SCORE = {score:.4f}, HONEYPOT: " + ("❌️","✅")[int(score >= max_score)])

        if score >= max_score:
            continue

        yield channel, score
