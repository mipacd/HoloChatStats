// frontend/src/pages/Home.tsx
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { FaGithub, FaXTwitter, FaCloud } from "react-icons/fa6"
import { api } from "@/lib/api"
type Update = { date: string; message: string }
type PublicationProgress = {
  behind: boolean
  target_month: string
  remaining_chat_logs: number
  approximate: boolean
}
export default function Home() {
  const { t, i18n } = useTranslation()
  const [dateRange, setDateRange] = useState<string[] | null>(null)
  const [logCount, setLogCount] = useState<number | null>(null)
  const [msgCount, setMsgCount] = useState<number | null>(null)
  const [publication, setPublication] = useState<PublicationProgress | null>(null)
  const [updates, setUpdates] = useState<Update[] | null>(null)
  const [errors, setErrors] = useState<Record<string, boolean>>({})
  useEffect(() => {
    api.get("/get_date_ranges")
      .then((res) => setDateRange(res.data))
      .catch(() => setErrors((e) => ({ ...e, dateRange: true })))
    api.get("/get_number_of_chat_logs")
      .then((res) => setLogCount(res.data))
      .catch(() => setErrors((e) => ({ ...e, logCount: true })))
    api.get("/get_num_messages")
      .then((res) => setMsgCount(res.data))
      .catch(() => setErrors((e) => ({ ...e, msgCount: true })))
    api.get("/get_publication_progress")
      .then((res) => setPublication(res.data))
      // Supplemental status must not make the coverage card fail as a whole.
      .catch(() => undefined)
    api.get("/get_latest_updates")
      .then((res) => setUpdates(res.data))
      .catch(() => setErrors((e) => ({ ...e, updates: true })))
  }, [])
  const featured = [
    { href: "/channel_clustering", img: "/cluster_snap.jpg", label: t("User Similarity Graph") },
    { href: "/membership_counts", img: "/member_snap.jpg", label: t("Membership Counts") },
    { href: "/user_info", img: "/user_info_snap.jpg", label: t("User Message Counts") },
  ]
  return (
    <div className="flex flex-col gap-8">
      {/* Hero */}
      <div className="bg-brand text-brand-foreground rounded-lg text-center py-14 px-5">
        <h1 className="text-3xl font-bold mb-3">{t("Welcome to HoloChatStats!")}</h1>
        <p className="text-lg text-muted-foreground mb-4">
          {t("Explore statistics and insights for Hololive and Indie VTuber chats.")}
        </p>
        <div className="flex justify-center gap-2 flex-wrap">
          <Button variant="secondary" asChild>
            <a href="https://github.com/mipacd/HoloChatStats" target="_blank" rel="noreferrer">
                <FaGithub className="mr-2 h-4 w-4" /> {t("GitHub")}
            </a>
            </Button>
            <Button asChild>
            <a href="https://twitter.com/HoloChatStat" target="_blank" rel="noreferrer">
                <FaXTwitter className="mr-2 h-4 w-4" /> {t("Twitter/X")}
            </a>
            </Button>
            <Button variant="outline" asChild>
            <a href="https://bsky.app/profile/holochatstats.info" target="_blank" rel="noreferrer">
                <FaCloud className="mr-2 h-4 w-4" /> {t("BlueSky")}
            </a>
            </Button>
        </div>
      </div>
      {/* Featured */}
      <Card>
        <CardHeader><CardTitle className="text-center">{t("Featured")}</CardTitle></CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-5">
            {featured.map((f) => (
              <a key={f.href} href={f.href} className="relative block rounded-lg overflow-hidden group">
                <img src={f.img} alt={f.label} className="w-full h-36 object-cover" />
                <span className="absolute bottom-0 left-0 w-full p-2 bg-black/70 text-white text-sm">
                  {f.label}
                </span>
              </a>
            ))}
          </div>
        </CardContent>
      </Card>
      {/* Stats + updates */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 max-w-4xl mx-auto w-full">
        <Card>
          <CardHeader><CardTitle className="text-center">{t("Data Coverage")}</CardTitle></CardHeader>
          <CardContent className="text-center space-y-2">
            {dateRange ? (
              <>
                <p><strong>{t("Start:")}</strong> {new Date(dateRange[0]).toUTCString().split(" GMT")[0]} UTC</p>
                <p><strong>{t("End:")}</strong> {new Date(dateRange[1]).toUTCString().split(" GMT")[0]} UTC</p>
              </>
            ) : errors.dateRange ? (
              <p className="text-destructive">{t("Unable to fetch date ranges.")}</p>
            ) : (
              <p className="text-muted-foreground">{t("Loading date ranges...")}</p>
            )}
            {logCount != null ? (
              <p><strong>{t("Number of Chat Logs Processed:")}</strong> {logCount.toLocaleString()}</p>
            ) : (
              <p className="text-muted-foreground">{t("Loading chat log count...")}</p>
            )}
            {msgCount != null ? (
              <p><strong>{t("Number of Chat Messages Processed:")}</strong> {msgCount.toLocaleString()}</p>
            ) : (
              <p className="text-muted-foreground">{t("Loading message count...")}</p>
            )}
            {publication?.behind && publication.target_month ? (
              <p className="pt-2 text-amber-700 dark:text-amber-400">
                {t("Approximately {{count}} chat logs remain before {{month}} is published.", {
                  count: publication.remaining_chat_logs.toLocaleString(i18n.resolvedLanguage),
                  month: new Intl.DateTimeFormat(i18n.resolvedLanguage, {
                    month: "long",
                    year: "numeric",
                    timeZone: "UTC",
                  }).format(new Date(`${publication.target_month}T00:00:00Z`)),
                })}
              </p>
            ) : null}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-center">{t("Latest Updates")}</CardTitle></CardHeader>
          <CardContent className="text-center space-y-2">
            {updates && updates.length > 0 ? (
              updates.map((u, i) => (
                <p key={i}><strong>{u.date}</strong>: {u.message}</p>
              ))
            ) : (
              <p className="text-muted-foreground">{t("No updates available.")}</p>
            )}
          </CardContent>
        </Card>
      </div>
      <Card className="max-w-4xl mx-auto w-full">
        <CardContent className="text-center pt-6">
          <p>
            If you think this site is impressive and you know of any software engineering
            opportunities,{" "}
            <a href="mailto:admin@holochatstats.info" className="underline">
              drop a message here
            </a>. Thanks!
          </p>
        </CardContent>
      </Card>
    </div>
  )
}
