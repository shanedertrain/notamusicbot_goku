import asyncio
import sys

import audio_generator as ag
import configuration as cfg
import newsapi as na

sys.path.append(str(cfg.FOLDER_ROOT / "rvc_cli"))
from rvc_cli import models

selection = 1

if selection == 0:
    media_name = "article"
    media_audio_path = asyncio.run(
        ag.generate_news_article_audio_file("OBAMA", media_name, category=na.Category.SPORTS)
    )
elif selection == 1:
    text = """
        Mr. L, thank you for having me on your show. I'm a big fan of your work, and I'm honored to be here.
        Now, about your question... if I had to survive 24 hours being chased by a horror villain of my choice to win 3 billion dollars, I would choose... Freddy Krueger.
        I know, I know, he's one of the most iconic horror villains of all time. But here's the thing: Freddy Krueger is a creature of dreams. 
        And if there's one thing I know about myself, it's that I can dream.
        I would use my dreams to my advantage, luring Freddy Krueger into a false sense of security. 
        I would make him think he had me cornered, only to have him wake up to find me gone. 
        I would keep him running in circles all night long, until the sun came up and he was forced to retreat.
        And then, I would collect my 3 billion dollars and use it to make the world a better place.
        What do you think, Mr. L? Do you think I have what it takes to survive 24 hours being chased by Freddy Krueger?
    """
    model = models.get_model("OBAMA")
    media_audio_path = asyncio.run(
        ag.generate_voice_converter_audio(model, text, cfg.FOLDER_OUTPUT / "manual_generation.mp3")
    )
