import { useCallback, useEffect, useMemo, useState } from "react"
import { Link, useNavigate, useParams } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { ArrowLeft, ExternalLink, Loader2, Search } from "lucide-react"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"

type ChannelOption = { channel_id: string; channel_name: string; channel_group: string }
type VideoItem = {
  video_id: string; title: string; channel_name: string; channel_group: string
  end_time: string; duration_seconds: number; message_count: number
  unique_chatters: number; member_chatters: number; member_percentage: number
  thumbnail_url: string
}
type Detail = {
  video: Pick<VideoItem, "video_id" | "title" | "channel_name" | "channel_group" |
    "end_time" | "duration_seconds" | "thumbnail_url"> &
    { channel_id: string; start_time: string; youtube_url: string }
  summary: { message_count: number; unique_chatters: number; member_chatters: number
    member_percentage: number; messages_per_minute: number | null; first_message_at: string | null }
  category_counts: Record<string, number>
  membership_rank_counts: Record<string, number>
  histogram: { bin_seconds: number; counts: number[] }
  funny_moments: { offset_seconds: number; start_seconds: number; count: number }[]
  word_counts: [string, number][]
}

const fmtDuration = (seconds: number) => {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`
    : `${m}:${String(s).padStart(2, "0")}`
}
const extractVideoId = (value: string) => {
  const direct = value.trim().match(/^[A-Za-z0-9_-]{11}$/)?.[0]
  if (direct) return direct
  try {
    const url = new URL(value.trim())
    const candidate = url.hostname === "youtu.be" ? url.pathname.slice(1) : url.searchParams.get("v")
    return candidate?.match(/^[A-Za-z0-9_-]{11}$/)?.[0] ?? null
  } catch { return null }
}

function SummaryCard({ label, value }: { label: string; value: string | number }) {
  return <Card><CardContent className="pt-5 text-center">
    <div className="text-2xl font-semibold">{value}</div>
    <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
  </CardContent></Card>
}

type TooltipEntry = { name?: string; value?: number; color?: string }
function DarkChartTooltip({ active, label, payload }: {
  active?: boolean; label?: string | number; payload?: TooltipEntry[]
}) {
  if (!active || !payload?.length) return null
  return <div className="min-w-28 rounded-lg border border-border bg-popover px-3 py-2 text-xs text-popover-foreground shadow-xl">
    {label != null && <p className="mb-1 font-medium">{label}</p>}
    {payload.map((entry, index) => <div key={`${entry.name}-${index}`} className="flex items-center justify-between gap-4">
      <span className="flex items-center gap-1.5 text-muted-foreground">
        <span className="size-2 rounded-full" style={{ backgroundColor: entry.color || "#4f8cff" }} />
        {entry.name || "Count"}
      </span>
      <span className="font-mono text-popover-foreground">{Number(entry.value || 0).toLocaleString()}</span>
    </div>)}
  </div>
}

type CloudWord = { word: string; count: number; x: number; y: number; size: number; rotate: number; color: string }
function layoutWordCloud(words: [string, number][]): CloudWord[] {
  if (!words.length) return []
  const width = 1000, height = 520
  const max = words[0][1] || 1
  const placed: (CloudWord & { left: number; right: number; top: number; bottom: number })[] = []
  const colors = ["#60a5fa", "#4ade80", "#f87171", "#c084fc", "#fbbf24", "#22d3ee"]
  for (const [word, count] of words) {
    const hash = [...word].reduce((value, char) => ((value * 33) ^ char.charCodeAt(0)) >>> 0, 5381)
    const size = 14 + 48 * Math.sqrt(count / max)
    const rotate = hash % 7 === 0 ? -28 : hash % 11 === 0 ? 28 : 0
    const radians = Math.abs(rotate) * Math.PI / 180
    const rawWidth = Math.max(size, word.length * size * 0.56)
    const rawHeight = size * 1.15
    const boxWidth = Math.abs(rawWidth * Math.cos(radians)) + Math.abs(rawHeight * Math.sin(radians))
    const boxHeight = Math.abs(rawWidth * Math.sin(radians)) + Math.abs(rawHeight * Math.cos(radians))
    for (let attempt = 0; attempt < 1400; attempt++) {
      const angle = attempt * 0.48 + (hash % 360) * Math.PI / 180
      const radius = 8.5 * Math.sqrt(attempt)
      const x = width / 2 + Math.cos(angle) * radius
      const y = height / 2 + Math.sin(angle) * radius * 0.68
      const box = { left: x - boxWidth / 2 - 3, right: x + boxWidth / 2 + 3,
        top: y - boxHeight / 2 - 2, bottom: y + boxHeight / 2 + 2 }
      if (box.left < 4 || box.right > width - 4 || box.top < 4 || box.bottom > height - 4) continue
      if (placed.some(other => !(box.right < other.left || box.left > other.right ||
        box.bottom < other.top || box.top > other.bottom))) continue
      placed.push({ word, count, x, y, size, rotate, color: colors[hash % colors.length], ...box })
      break
    }
  }
  return placed
}

function Browse() {
  const { t } = useTranslation()
  const navigate = useNavigate()
  const [months, setMonths] = useState<string[]>([])
  const [channels, setChannels] = useState<ChannelOption[]>([])
  const [month, setMonth] = useState("")
  const [group, setGroup] = useState("all")
  const [channel, setChannel] = useState("all")
  const [query, setQuery] = useState("")
  const [search, setSearch] = useState("")
  const [items, setItems] = useState<VideoItem[]>([])
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(0)
  const [total, setTotal] = useState(0)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    api.get("/stream-stats/options").then(({ data }) => {
      setMonths(data.months || [])
      setChannels(data.channels || [])
      setMonth((current) => current || data.months?.[0] || "")
    }).finally(() => setLoading(false))
  }, [])
  const groups = useMemo(() => Array.from(new Set(channels.map(c => c.channel_group))).sort(), [channels])
  const visibleChannels = useMemo(() => channels.filter(c => group === "all" || c.channel_group === group), [channels, group])
  const load = useCallback(() => {
    if (!month) return
    setLoading(true)
    api.get("/stream-stats", { params: { month, page, page_size: 25,
      ...(group !== "all" ? { group } : {}),
      ...(channel !== "all" ? { channel } : {}), ...(search ? { q: search } : {}) } })
      .then(({ data }) => { setItems(data.items || []); setPages(data.pages || 0); setTotal(data.total || 0) })
      .catch(() => { setItems([]); setPages(0); setTotal(0) })
      .finally(() => setLoading(false))
  }, [month, group, channel, search, page])
  useEffect(load, [load])
  useEffect(() => { setPage(1); setChannel("all") }, [group])
  useEffect(() => { setPage(1) }, [month, channel, search])
  useEffect(() => {
    registerEriContext(() => ({ page: "stream_stats", endpoint: "/api/stream-stats",
      parameters: { month, group, channel }, description: `Browsing aggregate stream statistics for ${month}` }))
    return () => registerEriContext(null)
  }, [month, group, channel])
  const submit = (event: React.FormEvent) => {
    event.preventDefault()
    const id = extractVideoId(query)
    if (id) navigate(`/stream_stats/${id}`)
    else setSearch(query.trim())
  }
  return <div className="space-y-6">
    <div className="text-center"><h1 className="text-3xl font-bold">{t("Per-Stream Statistics")}</h1></div>
    <Card><CardContent className="pt-6"><div className="grid gap-4 md:grid-cols-3 lg:grid-cols-4">
      <div><Label>{t("Month:")}</Label><Select value={month} onValueChange={setMonth}><SelectTrigger><SelectValue /></SelectTrigger>
        <SelectContent>{months.map(m => <SelectItem key={m} value={m}>{m}</SelectItem>)}</SelectContent></Select></div>
      <div><Label>{t("Group:")}</Label><Select value={group} onValueChange={setGroup}><SelectTrigger><SelectValue /></SelectTrigger>
        <SelectContent><SelectItem value="all">{t("All groups")}</SelectItem>{groups.map(g => <SelectItem key={g} value={g}>{g}</SelectItem>)}</SelectContent></Select></div>
      <div><Label>{t("Channel:")}</Label><Select value={channel} onValueChange={setChannel}><SelectTrigger><SelectValue /></SelectTrigger>
        <SelectContent><SelectItem value="all">{t("All channels")}</SelectItem>{visibleChannels.map(c => <SelectItem key={c.channel_id} value={c.channel_id}>{c.channel_name}</SelectItem>)}</SelectContent></Select></div>
      <form onSubmit={submit}><Label>{t("Find a stream")}</Label><div className="flex gap-2"><Input value={query} onChange={e => setQuery(e.target.value)} placeholder={t("Title, video ID, or YouTube URL")} /><Button type="submit" size="icon"><Search /></Button></div></form>
    </div></CardContent></Card>
    <div className="text-sm text-muted-foreground">{t("Streams found")}: {total.toLocaleString()}</div>
    {loading ? <Loader2 className="mx-auto h-10 w-10 animate-spin" /> :
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">{items.map(item =>
        <Link key={item.video_id} to={`/stream_stats/${item.video_id}`}><Card className="h-full transition-colors hover:bg-muted/40 overflow-hidden">
          <img src={item.thumbnail_url} alt="" className="aspect-video w-full object-cover" loading="lazy" />
          <CardHeader className="pb-2"><CardTitle className="line-clamp-2 text-base">{item.title}</CardTitle></CardHeader>
          <CardContent className="text-sm text-muted-foreground"><div>{item.channel_name}</div><div>{new Date(item.end_time).toLocaleString()}</div>
            <div className="mt-2 flex justify-between"><span>{item.message_count.toLocaleString()} {t("messages")}</span><span>{item.unique_chatters.toLocaleString()} {t("chatters")}</span></div></CardContent>
        </Card></Link>)}</div>}
    {!loading && !items.length && <p className="py-10 text-center text-muted-foreground">{t("No stream statistics found.")}</p>}
    <div className="flex justify-center items-center gap-3"><Button variant="outline" disabled={page <= 1} onClick={() => setPage(p => p - 1)}>{t("Previous")}</Button>
      <span>{page} / {Math.max(1, pages)}</span><Button variant="outline" disabled={page >= pages} onClick={() => setPage(p => p + 1)}>{t("Next")}</Button></div>
  </div>
}

function Details({ videoId }: { videoId: string }) {
  const { t } = useTranslation()
  const [data, setData] = useState<Detail | null>(null)
  const [loading, setLoading] = useState(true)
  const [missing, setMissing] = useState(false)
  useEffect(() => { setLoading(true); setMissing(false); setData(null); api.get(`/stream-stats/${videoId}`).then(r => setData(r.data))
    .catch(() => setMissing(true)).finally(() => setLoading(false)) }, [videoId])
  useEffect(() => {
    registerEriContext(() => ({ page: "stream_stats_detail", endpoint: `/api/stream-stats/${videoId}`,
      parameters: { video_id: videoId }, description: `Viewing aggregate statistics for YouTube stream ${videoId}` }))
    return () => registerEriContext(null)
  }, [videoId])
  const cloud = useMemo(() => layoutWordCloud(data?.word_counts || []), [data])
  if (loading) return <Loader2 className="mx-auto mt-20 h-10 w-10 animate-spin" />
  if (missing || !data) return <div className="text-center space-y-4"><h1 className="text-2xl font-bold">{t("Stream statistics unavailable")}</h1><Button asChild><Link to="/stream_stats">{t("Back to streams")}</Link></Button></div>
  const hist = data.histogram.counts.map((count, index) => ({ count, offset: index * data.histogram.bin_seconds, label: fmtDuration(index * data.histogram.bin_seconds) }))
  const categories = Object.entries(data.category_counts).map(([name, count]) => ({ name, count }))
  const ranks = Object.entries(data.membership_rank_counts).map(([rank, count]) => ({ name: rank === "-1" ? t("Non-members") : rank === "-2" ? t("Gift only") : Number(rank) === 0 ? t("New members") : `${rank} ${t("months")}`, count }))
  const openAt = (seconds: number) => window.open(`${data.video.youtube_url}&t=${Math.max(0, Math.floor(seconds))}s`, "_blank", "noopener")
  return <div className="space-y-8">
    <Button variant="ghost" asChild><Link to="/stream_stats"><ArrowLeft /> {t("Back to streams")}</Link></Button>
    <div className="grid gap-6 md:grid-cols-[minmax(280px,480px)_1fr]"><a href={data.video.youtube_url} target="_blank" rel="noreferrer"><img src={data.video.thumbnail_url} alt="" className="w-full rounded-lg" /></a>
      <div><h1 className="text-2xl font-bold">{data.video.title}</h1><p className="text-muted-foreground">{data.video.channel_name} · {new Date(data.video.end_time).toLocaleString()}</p>
        <p className="mt-2">{t("Duration")}: {fmtDuration(data.video.duration_seconds)}</p><Button className="mt-4" asChild><a href={data.video.youtube_url} target="_blank" rel="noreferrer">{t("Watch on YouTube")} <ExternalLink /></a></Button></div></div>
    <div className="grid grid-cols-2 gap-3 lg:grid-cols-5"><SummaryCard label={t("Chat messages")} value={data.summary.message_count.toLocaleString()} /><SummaryCard label={t("Avg. message rate")} value={data.summary.messages_per_minute == null ? "—" : data.summary.messages_per_minute.toLocaleString()} /><SummaryCard label={t("Unique chatters")} value={data.summary.unique_chatters.toLocaleString()} /><SummaryCard label={t("Members")} value={data.summary.member_chatters.toLocaleString()} /><SummaryCard label={t("Member percentage")} value={`${data.summary.member_percentage}%`} /></div>
    <div className="grid gap-6 lg:grid-cols-2">{[[t("Message categories"), categories], [t("Membership ranks"), ranks]].map(([title, rows]) => <Card key={title as string}><CardHeader><CardTitle>{title as string}</CardTitle></CardHeader><CardContent className="h-72"><ResponsiveContainer><BarChart data={rows as {name:string,count:number}[]} layout="vertical"><CartesianGrid strokeDasharray="3 3" /><XAxis type="number" /><YAxis dataKey="name" type="category" width={110} /><Tooltip content={<DarkChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.08)" }} /><Bar dataKey="count" fill="#4f8cff" /></BarChart></ResponsiveContainer></CardContent></Card>)}</div>
    <Card><CardHeader><CardTitle>{t("Chat velocity")}</CardTitle></CardHeader><CardContent><p className="mb-3 text-sm text-muted-foreground">{t("Messages per minute. Click a bar to open YouTube at that point.")}</p><div className="h-72 overflow-x-auto"><div style={{ minWidth: Math.max(700, hist.length * 10), height: "100%" }}><ResponsiveContainer><BarChart data={hist}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="label" minTickGap={30} /><YAxis /><Tooltip content={<DarkChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.08)" }} /><Bar dataKey="count" fill="#4f8cff" onClick={(row) => { const point = row.payload as { offset: number }; openAt(point.offset) }} className="cursor-pointer" /></BarChart></ResponsiveContainer></div></div></CardContent></Card>
    <Card><CardHeader><CardTitle>{t("Funny moments")}</CardTitle></CardHeader><CardContent className="flex flex-wrap gap-2">{data.funny_moments.length ? data.funny_moments.map(m => <Button key={m.offset_seconds} variant="outline" onClick={() => openAt(m.start_seconds)}>{fmtDuration(m.offset_seconds)} · {m.count}</Button>) : <span className="text-muted-foreground">{t("No funny moments detected.")}</span>}</CardContent></Card>
    <Card><CardHeader><CardTitle>{t("Word cloud")}</CardTitle></CardHeader><CardContent><p className="mb-4 text-sm text-muted-foreground">{t("Only languages with clear word boundaries are included.")}</p><div className="overflow-hidden rounded-lg bg-muted/20"><svg viewBox="0 0 1000 520" className="block min-h-72 w-full" role="img" aria-label={t("Word cloud")}>{cloud.map(item => <text key={item.word} x={item.x} y={item.y} textAnchor="middle" dominantBaseline="middle" fill={item.color} fontSize={item.size} fontWeight={item.size > 38 ? 650 : 500} transform={`rotate(${item.rotate} ${item.x} ${item.y})`}><title>{`${item.word}: ${item.count}`}</title>{item.word}</text>)}</svg></div></CardContent></Card>
  </div>
}

export default function StreamStats() {
  const { videoId } = useParams()
  return videoId ? <Details videoId={videoId} /> : <Browse />
}
