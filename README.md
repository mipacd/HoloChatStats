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
    * Channel Clusters by Common Members: Generates a node graph for a given month showing which channels have similar user patterns. The connections between nodes are calculated using cosine similarity, 
    with a threshold of 80th, 90th, or 95th percentile of edge weights. Communities are shown as the colors of the nodes, and are calculated using Leiden community detection. This data is calculated from the number of chat messages each user has made 
    in each channel in a given month. Placement of the nodes and any numbers that appear on the graph are dynamic and not part of the calculation. Channels must meet a similarity threshold with at least one other channel to appear on the graph.
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
* Funniest Moments: Uses feature detection to algorithmically determine the "funniest" moment of each stream, based on the greatest concentration of humorous reactions (lol, lmao, etc.) in each stream. This timestamp is shifted back 10 seconds to provide a leadup to the moment. 
* LLM Insights: An LLM query interface is available for select graphs (membership gain/loss, membership counts, user gain/loss, chat makeup). This interface is restricted to 3 queries per user per day and only 
English is supported. Access to the LLM interface is blocked in Japan and South Korea due to a combination of the language restriction, the significant number of site users from these countries, and the limited 
number of free daily queries from OpenRouter (50 per day).
* CSV Downloads
* Containerization with Docker
* Support for every Hololive member and select indies (currently Nimi, Dooby, Dokibird, Mint, Sakuna, Rica, Ruka)

## Planned Features / TODO (subject to change)

* Public API (depending on available hosting): This feature and the one below it would require site donations and/or a paid service.
* Mobile App (dependant on public API)
* Attrition rates of graduated/affiliated talent's fanbases from all Hololive chats (API Done)
* Data ingest optimizations
* Unit tests

## Data Ingestion Setup

1. Setup a PostgreSQL database (default: youtube_data).

2. Obtain a YouTube Data v3 API key.

3. Export your YouTube cookies in Netscape format and save them into data_ingestion.

4. Rename data_ingestion/config.ini.sample to data_ingestion/config.ini and edit this file to include the above information.

5. Deploy the included Dockerfile (optional) and configure the script to run at a regular interval using cron or other scheduler

6. If not using Docker, run `pip install requirements.txt`.

7. `downloader.py` takes two arguments: --month and --year. If these arguments are not specifed, the script will run against the current month. If it is the first of the month, 
the script will run against the previous month.

8. The data ingestion process can take some time, about an hour or more for one day in order to calculate the materialized views.

## Web Server Setup

1. Rename web/config.ini.sample to web/config.ini and set variables.

2. Obtain LLM API key from OpenRouter and Google Analytics keys (optional)

3. Deploy using Docker or install requirements.txt and run server.py

## Contributing 
Contributions are welcome (backend/frontend/ML devs, translators/i18n experience, UI/UX, etc.)

## Support This Project

If you enjoy this project, consider supporting its development:

-  [Donate on Ko-Fi](https://ko-fi.com/holochatstats)  Your support helps keep this project alive and growing.

Looking to help in other ways?  
- **Hosting sponsorships**: Interested in hosting this project? Let's collaborate!  
- **Employment opportunities**: If you’re looking for a passionate developer, feel free to reach out.

📧 Contact me: [admin@holochatstats.info](mailto:admin@holochatstats.info)
