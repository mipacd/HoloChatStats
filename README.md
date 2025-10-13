# HoloChatStats

HoloChatStats is a platform for providing YouTube live chat language and user data, as well as streaming freqency data, for VTubers.

This is a replacement (v2) for the original holochatstats.info site that launched in January 2021 and as a replacement for HoloAutoClip (clip.holochatstats.info) that launched in November 2020.
HoloChatStats v1 will continue to be available as an archive at old.holochatstats.info.

All data is derived from a combination of publicly available chat logs and metadata from the YouTube API.

## Current Features
* Membership Information
    * Member Counts: The number of members detected for each channel, by badge type. Some channels may lack certain badges. 
    * Member Percentages: The percentage of total chat users during a given month with a membership to a channel.
    * Membership Gain/Loss: The number of chat users who had a membership in the given month and not the previous month, and vice versa.
* Langauge Statistics
    * Chat Makeup: The average message rate per minute of certain message types in the given month. Message types are: JP, KR, RU, Emote, and EN/ES/ID/etc.
    JP, KR, and RU message types are detected by character set matching. Emotes messages are defined as messages that are entirely composed of Unicode or YouTube membership emotes.
    Messages that do not meet any of the previous classifications are classified as EN/ES/ID/etc.
    * Memberships and Rates by Language: The percentage and average rate of each of the above language classifications by channel over time. Emote message counts are subtracted from the
    total message count while calculating percentages. Total duration is computed using the total number of streaming hours for all available chat logs.
    * JP User Percentages: The percentage of users of a given channel that have sent JP lanaguage chat messages over 50% of the time across all channels for a given month (excluding emote messages).
* User Data
    * Common User Percentages by Channel / Month: Given two channel/month pairs (A and B), determine the number of common users between the pairs. Also, calculate the percent of chat users in A that
    also chatted in B, and vice versa.
    * Common Member Percentages by Channel / Month: Same as above for chat users holding a membership.
    * Common User Heatmap: Displays a heatmap showing the percentage of common users (or members) between channels for a given month. 
    * Channel Clusters by Common Members: Generates a node graph for a given month showing which channels have similar user patterns. The connections between nodes are calculated using cosine similarity, 
    with a threshold of 80th, 90th, or 95th percentile of edge weights. Communities are shown as the colors of the nodes, and are calculated using Leiden community detection. This data is calculated from the number of chat messages each user has made 
    in each channel in a given month. Node placement is dynamic and not tied to the dataset. Channels must meet a similarity threshold with at least one other channel to appear on the graph.
    * Active User Gain / Loss: The number of chat users that met a 5 message threshold in the specified month and not the previous month (gain) and vice versa (loss). Channels that did not stream in either month 
    will not be shown on the graph.
    * Chat Leaderborads: Displays the top 10 chat users by message count for a given channel and month.
    * Exclusive Chat Users: The percentage of a channels chat users that did not participate in any other channel's chat within the channel's assigned group (e.g. Hololive, Indie). 
    * Message Frequencies by User: Given a username and a month, display counts and percentiles for each channel's chat that the user participated in.
    * Engagement Rates: Average number of messages per user by channel for a given month.
    * Recommendation Engine: Given a username, determines the top 5 channels that other users with similar chat patterns have participated in, excluding any channels that the user has participated in, during the previous calendar month. This is calculated using cosine similarity, using a similar method to the channel clusters graph.
* Streaming Hours (all streaming hours are calculated from archived, public streams only)
    * Total Streaming Hours: Total streaming hours by channel in a given month
    * Average Streaming Hours: Average stream duration for each channel in a given month
    * Longest Streeam Duration: Duration of each channel's longest stream in a given month
    * Streaming Hour Change: The change in streaming hours from the previous month for each channel
    * Monthly Streaming Hours: The streaming hours for a given channel over time
* Highlights
    * Funniest Moments: Uses feature detection to algorithmically determine the "funniest" moment of each stream, based on the greatest concentration of humorous reactions (lol, lmao, etc.) in each stream. This timestamp is shifted back 10 seconds to provide a leadup to the moment. 
    * AI Summarized Highlights (BETA): Detects the top moments of each stream using chat velocity (number varies by stream duration). Summarizes those moments by passing a snippet of chat from around each timestamp into an LLM. 
    * Search AI Highlights (BETA): Searches the LLM generated text using vector search. Supports "channel:", "from:", and "to:" search engine operators.
* LLM Insights: An LLM query interface is available for select graphs (membership gain/loss, membership counts, user gain/loss, chat makeup). This interface is restricted to 3 queries per user per day and only 
English is supported. Access to the LLM interface is blocked in Japan and South Korea due to a combination of the language restriction, the significant number of site users from these countries, and the limited 
number of free daily queries from OpenRouter (50 per day).
* CSV Downloads
* Containerization with Docker
* Support for every Hololive member and select indies (currently Nimi, Saba, Dooby, Dokibird, Mint, Sakuna, Rica, Ruka, Roa, Rei)

## Planned Features / TODO (subject to change)

* Public API (possible paid feature in the future)
* Mobile App (dependant on demand)
* Unit tests
* Kubernetes Deployment
* Trend forcasting / Sentiment Analysis (PyTorch and/or TensorFlow)

## Web Server Setup

1. Install Docker and Docker Compose (https://www.docker.com/)

2. Check out the repo: `git clone https://github.com/mipacd/HoloChatStats`

3. Rename .env.sample to .env and configure settings. LLM chart queries require an [OpenRouter](https://openrouter.ai/) API key. Some models can be used for free (with a limited daily quota). By default, HoloChatStats limits users to 3 queries per day. Rename web/news.txt.sample to web/news.txt (enter site news in this file using the format shown in the sample).

4. Bring up the stack (web, PostgreSQL, Redis) with `docker compose up -d` from the repo's root directory.

5. Site will be available on port 80. If you are running on your local machine, it will be available at http://localhost.

6. Run `docker compose down` to bring down the stack.

## LLM Server Setup (optional)

1. Put the contents of the llm_server directory on a machine with a higher-end GPU (recommended: RTX 3080 or better, 16 GB+ VRAM, only NVIDIA GPUs are supported).

2. In the llm_server directory, run `pip install -r requirements.txt`

3. Install [CUDA](https://developer.nvidia.com/cuda-downloads).

4. Install [Ollama](https://ollama.com/).

5. Pull the model with Ollama: `ollama pull deepseek-r1:7b-qwen-distill-q4_K_M`

6. Run the server: `uvicorn main:app --port 8000`


## Data Ingestion Setup

1. Enter the data_ingestion directory and run `pip install -r requirements.txt`

2. Rename config.sample.ini to config.ini. Enter your configuration in this file. Data ingestion needs a YouTube Data v3 API key (available for free with a quota using the [Google Cloud Console](https://console.cloud.google.com/apis/library/youtube.googleapis.com)). Set DBHost to the IP or hostname of the machine running the web server (or leave as localhost if it is the same machine). Set the AIServerURL to the LLM server host, if used.

3. Run `python downloader.py <ARGS>`

    Arguments:

    ``--month <INT>`` ``--year <INT>``: Month and year to process. Optional. Defaults to the previous month if run on the first of the month and the current month if run on the 2nd of the month or later.

    ``--disable_ai_summarization`` OR ``-d``: Disables AI summarization of chat snippets. Always run with this flag if you are not running the LLM server.

4. Optional: Add to cron or Windows Task Scheduler to run nightly.

## Contributing 
Contributions are welcome (backend/frontend/ML devs, translators/i18n experience, UI/UX, etc.)

## Support This Project

If you enjoy this project, consider supporting its development:

-  [Donate on Ko-Fi](https://ko-fi.com/holochatstats)  Your support helps keep this project alive and growing.

Looking to help in other ways?  
- **Hosting sponsorships**: Interested in hosting this project? Let's collaborate!  
- **Employment opportunities**: If you’re looking for a passionate developer, feel free to reach out.

📧 Contact me: [admin@holochatstats.info](mailto:admin@holochatstats.info)
