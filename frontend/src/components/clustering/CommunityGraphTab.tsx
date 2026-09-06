import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Info, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { MonthPicker } from "@/components/ui/month-picker"
import { CommunityGraphView } from "./CommunityGraphView"
import type { CommunityGraphData } from "./CommunityGraphView"
import { api } from "@/lib/api"
import { useElementSize } from "@/hooks/useElementSize"
import { registerEriContext } from "@/components/eri/eri-context"
export function CommunityGraphTab() {
  const { t, i18n } = useTranslation()
  const [month, setMonth] = useState("")
  const [group, setGroup] = useState("all")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [data, setData] = useState<CommunityGraphData | null>(null)
  const [loaded, setLoaded] = useState<{ month: string; group: string } | null>(null)
  const { ref: containerRef, width, height } = useElementSize<HTMLDivElement>()
  /* ── Eri context ─────────────────────────────────────────────────────── */
  useEffect(() => {
    registerEriContext(() => ({
      page: "community_graph",
      endpoint: "/api/community_graph",
      parameters: { month, channel_group: group },
      description: `Viewing community graph for ${month}`,
    }))
    return () => registerEriContext(null)
  }, [month, group])
  /* ── Fetch ───────────────────────────────────────────────────────────── */
  const loadGraph = async () => {
    if (!month) {
      setError(t("Please select a month."))
      return
    }
    setError(null)
    setLoading(true)
    try {
      const params: Record<string, string> = {
        month,
        include_edges: "true",
      }
      if (group !== "all") params.channel_group = group
      const res = await api.get("/community_graph", { params })
      if (res.data.error) {
        setError(res.data.error)
        setData(null)
        setLoaded(null)
        return
      }
      setData(res.data)
      setLoaded({ month, group })
    } catch {
      setError(t("Failed to load community graph."))
      setData(null)
      setLoaded(null)
    } finally {
      setLoading(false)
    }
  }
  return (
    <TooltipProvider>
      <div className="flex flex-col gap-4">
        {/* ── Header ─────────────────────────────────────────────────── */}
        <h2 className="text-2xl font-bold text-center flex items-center justify-center gap-2">
          {t("Community Graph")}
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-4 w-4 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              {t(
                "Bipartite graph of users and channels with Leiden community detection. " +
                  "Channels are positioned by user-overlap similarity. Each user dot is " +
                  "placed near its most-active channels. Scroll to zoom, drag to pan."
              )}
            </TooltipContent>
          </Tooltip>
        </h2>
        {/* ── Controls ───────────────────────────────────────────────── */}
        <div className="flex flex-wrap items-end justify-center gap-4">
          <div className="space-y-1.5">
            <Label>{t("Select Month:")}</Label>
            <MonthPicker value={month} onChange={setMonth} locale={i18n.language} />
          </div>
          <div className="space-y-1.5">
            <Label>{t("Group:")}</Label>
            <Select value={group} onValueChange={setGroup}>
              <SelectTrigger className="w-[160px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">{t("All")}</SelectItem>
                <SelectItem value="Hololive">{t("Hololive")}</SelectItem>
                <SelectItem value="Indie">{t("Indie")}</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <Button onClick={loadGraph} disabled={loading}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {t("Load Graph")}
          </Button>
        </div>
        {/* ── Stats bar ──────────────────────────────────────────────── */}
        {data && (
          <p className="text-center text-sm text-muted-foreground">
            {data.stats.user_count.toLocaleString()} {t("users")} &middot;{" "}
            {data.stats.channel_count} {t("channels")} &middot;{" "}
            {data.stats.community_count} {t("communities")}
          </p>
        )}
        {error && <p className="text-center text-destructive text-sm">{error}</p>}
        {/* ── Canvas area ────────────────────────────────────────────── */}
        <div
          ref={containerRef}
          className="relative"
          style={{
            width: "95vw",
            marginLeft: "calc(50% - 47.5vw)",
            height: "calc(100vh - 280px)",
            minHeight: 400,
          }}
        >
          {data && loaded && width > 0 && height > 0 ? (
             <CommunityGraphView
               key={data.title}
               data={data}
               width={width}
               height={height}
              month={loaded.month}
              channelGroup={loaded.group}
             />
          ) : (
            loading && (
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
              </div>
            )
          )}
        </div>
      </div>
    </TooltipProvider>
  )
}