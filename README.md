# DiscordHoneypotDetector

**For educational purposes only.**
Due to recents discord servers adding a honepot channel to catch scammers, here is a AI powered detector. 
This is just the detector hitself, you will need to integrate it into your Discord script.

Check out usage.py to know how to use the ai (only the ai) or check out detector.py to use the detector function which is tailored to avoid false positives and add more checks as well as the ai model itself.

### Usage example
```python
# discord.py-self -> Python library to use a client as a bot
import discord
from discord.ext import commands
import os

import detector

bot = commands.Bot(command_prefix='>', self_bot=True)

@bot.event
async def on_ready():
    print("Ready")
    guild = bot.get_guild(1359831916072665099)
    if not guild:
        return
    async for channel, score in detector.find_safe_channels(bot, guild, debug=False):
        print(channel.name, score) # This will print every channel with a score lower than max_score (2.3 by default, check out detector.py)

bot.run(
    "Your token"
)

```

Current detector_model.pkl is trained on current dataset, feel free to modify anything you want, but note that it's pretty large for github.

By: Bob (t.me/arkaseon)
Feel free to dm me to add messages to datasets, modify the code, or something else.
