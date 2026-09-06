export interface PromptGroup {
  label: string
  prompts: string[]
}
export const samplePromptGroups: PromptGroup[] = [
  {
    label: "General Queries",
    prompts: [
      "Who streamed the most in Hololive last month?",
      "Which Hololive member had the highest average stream duration last month?",
      "Which Hololive member had the longest stream last month?",
      "Which Hololive member had the highest average Korean chat messages per minute last month?",
      "Which Hololive member had the most members last month?",
      "What are the top 3 Hololive members that gained the most memberships last month?",
      "Which Hololive member had the greatest increase in streaming hours last month?",
      "How can I support HoloChatStats?",
      "Hi Eri! How are you today? Who is your oshi?",
    ],
  },
  {
    label: "Templated",
    prompts: [
      "Recommend some VTubers for me. My channel ID is: [YOUR_CHANNEL_ID_HERE]",
      "How many hours did [VTUBER_NAME] stream last month?",
      "How many chat users were common to [VTUBER_NAME] and [VTUBER_NAME] last month?",
      "How many membered chat users were common to [VTUBER_NAME] and [VTUBER_NAME] last month?",
      "Who are the most active chatters for [VTUBER_NAME] last month?",
      "What percentage of chat users chatted exclusively in [VTUBER_NAME]'s chat last month?",
      "What percentage of [VTUBER_NAME]'s chat was in Japanese last month?",
      "What percentage of [GRADUATED_FANBASE_NAME] still participate in Hololive chats?",
      "Give me some YouTube links to funny moments from [VTUBER_NAME]'s streams last month.",
      "Can you tell me which channels I chatted on last month? My channel ID is: [YOUR_CHANNEL_ID_HERE]",
      "Show me Hololive merch currently on sale for [VTUBER_NAME]",
      "How many subscribers does [VTUBER_NAME] have?",
      "What was [VTUBER_NAME]'s last stream?",
    ],
  },
]