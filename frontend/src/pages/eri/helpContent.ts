import {
  Video, MessageSquare, Sparkles, ShoppingBag, CircleUser, PanelRight,
  type LucideIcon,
} from "lucide-react"
export interface HelpItem {
  icon?: LucideIcon
  title: string
  body: string | string[]
}
export interface HelpSection {
  header: string
  items: HelpItem[]
}
export const helpSections: HelpSection[] = [
  {
    header: "Eri's Capabilities",
    items: [
      {
        icon: Video,
        title: "Stream Stats & Schedules",
        body: "Ask Eri about streaming habits (monthly hours, averages, longest streams) or check real-time schedules to see who is live, upcoming, or recently finished streaming.",
      },
      {
        icon: MessageSquare,
        title: "Chat & Community",
        body: "Analyze chat language makeup (EN/JP/KR/RU), engagement rates, and membership trends. Eri can also determine fanbase overlap by finding common users between different channels.",
      },
      {
        icon: Sparkles,
        title: "Content & Highlights",
        body: "Find AI-summarized highlights, search for specific moments (e.g., 'funny moments from Marine'), or locate the 'funniest' timestamp in a stream based on chat reactions.",
      },
      {
        icon: ShoppingBag,
        title: "Channel & Merch",
        body: "Get channel metrics like subscriber counts and total views. Eri can also search the official Hololive Shop for merchandise (English or Japanese).",
      },
      {
        icon: CircleUser,
        title: "Personal Statistics",
        body: "Provide your YouTube Channel ID (starts with 'UC...') to see your own chat frequency, percentiles, and get channel recommendations based on your history.",
      },
      {
        icon: PanelRight,
        title: "Page Context",
        body: "Eri has a widget on most dataset pages and can answer questions about the specific data you are viewing. However, she cannot 'see' the visual UI or charts exactly as you do.",
      },
    ],
  },
  {
    header: "Eri's Limitations",
    items: [
      {
        title: "Data Scope",
        body: [
          "Eri only has data for Hololive and select Indies. She does NOT have data for other agencies (e.g., Nijisanji) or Holostars.",
          "Chat stats are updated monthly. Real-time data is available for schedules and merch only. No SuperChat or membership stream data.",
        ],
      },
      {
        title: "Accuracy & Complexity",
        body: [
          "Eri is an AI and may occasionally provide inaccurate information or hallucinate data.",
          "She is limited to 3 data requests per prompt - please break down complex questions (e.g., comparing 5 different months).",
        ],
      },
      {
        title: "Daily Query Limits",
        body: "You have a limited number of queries per day, indicated by the counter next to the chat bubble icon. This quota resets daily at midnight UTC.",
      },
    ],
  },
]