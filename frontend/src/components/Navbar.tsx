import { Link } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Menu, ChevronDown } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Sheet, SheetContent, SheetTrigger } from "@/components/ui/sheet"
type LinkItem = { to: string; label: string; external?: boolean }
type DropdownGroup = { label: string; items: LinkItem[] }
const itemClass =
  "cursor-pointer focus:bg-white/10 focus:text-white data-[highlighted]:bg-white/10"
export function Navbar() {
  const { t, i18n } = useTranslation()
  const groups: DropdownGroup[] = [
    {
      label: t("Users and Memberships"),
      items: [
        { to: "/common_users_members", label: t("Common Users / Members") },
        { to: "/heatmap", label: t("Common User Heatmap") },
        { to: "/channel_clustering", label: t("User Similarity Graph") },
        { to: "/user_change", label: t("Active User Gain / Loss") },
        { to: "/chat_leaderboards", label: t("Chat Leaderboards") },
        { to: "/exclusive_chat", label: t("Exclusive Chat Users") },
        { to: "/engagement", label: t("Engagement Rates") },
        { to: "/user_info", label: t("Message Frequencies by User") },
        { to: "/membership_counts", label: t("Membership Counts") },
        { to: "/membership_percentages", label: t("Membership Percentages") },
        { to: "/membership_change", label: t("Membership Gain / Loss") },
        { to: "/recommendation_engine", label: t("Recommendation Engine") },
      ],
    },
    {
      label: t("Language Stats"),
      items: [
        { to: "/chat_makeup", label: t("Chat Makeup") },
        { to: "/message_types", label: t("Message Percentages and Rates by Language") },
        { to: "/jp_user_percents", label: t("JP User Percentages") },
      ],
    },
    {
      label: t("Streaming Hours"),
      items: [
        { to: "/streaming_hours", label: t("Total Streaming Hours") },
        { to: "/streaming_hours_avg", label: t("Average Streaming Hours") },
        { to: "/streaming_hours_max", label: t("Longest Stream Duration") },
        { to: "/streaming_hours_diff", label: t("Streaming Hour Change") },
        { to: "/monthly_streaming_hours", label: t("Monthly Streaming Hours") },
        { to: "/stream_freq", label: t("Stream Frequency Heatmap") },
      ],
    },
    {
      label: t("Content"),
      items: [
        { to: "/viewer", label: t("Live Stream Viewer") },
        { to: "/content_similarity", label: t("Content Similarity Graph") },
        { to: "/funniest_timestamps", label: t("Funniest Moments") },
      ],
    },
  ]
  const moreItems: LinkItem[] = [
    { to: "/site_metrics", label: t("Site Metrics") },
    { to: "https://holochatstats.wordpress.com", label: t("Blog"), external: true },
    { to: "https://old.holochatstats.info", label: "HoloChatStats v1", external: true },
  ]
  const languages = [
    { code: "en", label: "English", flag: "fi-us" },
    { code: "ja", label: "日本語", flag: "fi-jp" },
    { code: "ko", label: "한국어", flag: "fi-kr" },
  ]
  const renderItem = (item: LinkItem) =>
    item.external ? (
      <a href={item.to} target="_blank" rel="noreferrer" className="w-full">
        {item.label}
      </a>
    ) : (
      <Link to={item.to} className="w-full">
        {item.label}
      </Link>
    )
  return (
    <nav className="bg-brand border-b border-border">
      <div className="container mx-auto max-w-7xl flex h-16 items-center justify-between px-4">
        {/* Left: brand + main groups */}
        <div className="flex items-center gap-1">
          <Link to="/" className="flex items-center gap-2 font-semibold mr-2">
            <img src="/logo.png" alt="HoloChatStats Logo" width={25} height={25} />
            HoloChatStats
          </Link>
          <div className="hidden lg:flex items-center gap-1">
            <Button variant="ghost" asChild>
              <Link to="/">{t("Home")}</Link>
            </Button>
            {groups.map((group) => (
              <DropdownMenu key={group.label}>
                <DropdownMenuTrigger asChild>
                  <Button
                    variant="ghost"
                    className="group gap-1 data-[state=open]:bg-white/10"
                  >
                    {group.label}
                    <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="bg-secondary text-secondary-foreground border-border">
                  {group.items.map((item) => (
                    <DropdownMenuItem key={item.to} asChild className={itemClass}>
                      {renderItem(item)}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            ))}
            <Button variant="ghost" asChild>
              <Link to="/eri">{t("Ask Eri")}</Link>
            </Button>
          </div>
        </div>
        {/* Right: More dropdown + language */}
        <div className="hidden lg:flex items-center gap-2">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="ghost"
                className="group gap-1 data-[state=open]:bg-white/10"
              >
                {t("More")}
                <ChevronDown className="h-4 w-4 transition-transform group-data-[state=open]:rotate-180" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              align="end"
              className="bg-secondary text-secondary-foreground border-border"
            >
              {moreItems.map((item) => (
                <DropdownMenuItem key={item.to} asChild className={itemClass}>
                  {renderItem(item)}
                </DropdownMenuItem>
              ))}
              <DropdownMenuSub>
                <DropdownMenuSubTrigger className={itemClass}>{t("Demos")}</DropdownMenuSubTrigger>
                <DropdownMenuSubContent className="bg-secondary text-secondary-foreground border-border">
                  <DropdownMenuItem asChild className={itemClass}>
                    <Link to="/highlights">{t("AI Summarized Highlights")}</Link>
                  </DropdownMenuItem>
                  <DropdownMenuItem asChild className={itemClass}>
                    <Link to="/highlight_search">{t("Search AI Highlights")}</Link>
                  </DropdownMenuItem>
                </DropdownMenuSubContent>
              </DropdownMenuSub>
            </DropdownMenuContent>
          </DropdownMenu>
          <Select value={i18n.language} onValueChange={(l) => i18n.changeLanguage(l)}>
            <SelectTrigger className="w-[140px] bg-card">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {languages.map((l) => (
                <SelectItem key={l.code} value={l.code}>
                  <span className="flex items-center gap-2">
                    <span className={`fi ${l.flag}`} />
                    {l.label}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        {/* Mobile */}
        <Sheet>
          <SheetTrigger asChild>
            <Button variant="ghost" size="icon" className="lg:hidden">
              <Menu className="h-5 w-5" />
            </Button>
          </SheetTrigger>
          <SheetContent side="right" className="bg-brand text-brand-foreground w-72 overflow-y-auto">
            <div className="flex flex-col gap-2 mt-8">
              <Link to="/" className="py-2 font-medium">{t("Home")}</Link>
              {groups.map((group) => (
                <div key={group.label} className="border-t border-border pt-2">
                  <p className="text-sm font-semibold text-muted-foreground mb-1">{group.label}</p>
                  {group.items.map((item) => (
                    <Link key={item.to} to={item.to} className="block py-1.5 pl-2 text-sm">
                      {item.label}
                    </Link>
                  ))}
                </div>
              ))}
              <Link to="/eri" className="py-2 border-t border-border">{t("Ask Eri")}</Link>
              <div className="border-t border-border pt-2">
                {moreItems.map((item) =>
                  item.external ? (
                    <a key={item.to} href={item.to} target="_blank" rel="noreferrer" className="block py-1.5 text-sm">
                      {item.label}
                    </a>
                  ) : (
                    <Link key={item.to} to={item.to} className="block py-1.5 text-sm">
                      {item.label}
                    </Link>
                  )
                )}
              </div>
              <Select value={i18n.language} onValueChange={(l) => i18n.changeLanguage(l)}>
                <SelectTrigger className="mt-4"><SelectValue /></SelectTrigger>
                <SelectContent>
                  {languages.map((l) => (
                    <SelectItem key={l.code} value={l.code}>
                      <span className="flex items-center gap-2">
                        <span className={`fi ${l.flag}`} />
                        {l.label}
                      </span>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </SheetContent>
        </Sheet>
      </div>
    </nav>
  )
}
