import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Info, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { MonthPicker } from "@/components/ui/month-picker"
import {
  Tooltip, TooltipContent, TooltipProvider, TooltipTrigger,
} from "@/components/ui/tooltip"
import { api } from "@/lib/api"
import { downloadCSV } from "@/lib/chart-export"
import { registerEriContext } from "@/components/eri/eri-context"
interface VideoRow {
  video_id: string
  title: string
  timestamp: number
}
function toHHMMSS(seconds: number) {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  const s = Math.floor(seconds % 60)
  return [h, m, s].map((v) => String(v).padStart(2, "0")).join(":")
}
export default function FunniestTimestamps() {
  const { t, i18n } = useTranslation()
  const [channels, setChannels] = useState<string[]>([])
  const [channel, setChannel] = useState("")
  const [month, setMonth] = useState("")
  const [rows, setRows] = useState<VideoRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.get("/get_channel_names").then((res) => setChannels(res.data || []))
  }, [])
  useEffect(() => {
    registerEriContext(() => ({
      page: "funniest_timestamps",
      endpoint: "/api/get_funniest_timestamps",
      parameters: { channel, month },
      description: `Viewing funniest timestamps for ${channel} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [channel, month])
  const fetchData = async () => {
    if (!channel || !month) {
      setError(t("Please select a channel and a month."))
      return
    }
    setError(null)
    setLoading(true)
    setRows(null)
    try {
      const res = await api.get("/get_funniest_timestamps", { params: { channel, month } })
      if (!res.data || res.data.length === 0) {
        setError(t("No data available."))
        return
      }
      setRows(res.data)
    } catch {
      setError(t("Error fetching data. Please try again later."))
    } finally {
      setLoading(false)
    }
  }
  const handleCSV = async () => {
    if (!channel || !month) {
      setError(t("Please select a channel and a month before downloading."))
      return
    }
    try {
      const res = await api.get("/get_funniest_timestamps", { params: { channel, month } })
      if (!res.data || res.data.length === 0) {
        setError(t("No data available for export."))
        return
      }
      downloadCSV(
        "funniest_moments.csv",
        ["Title", "URL", "Timestamp"],
        res.data.map((v: VideoRow) => [
          v.title,
          `https://www.youtube.com/watch?v=${v.video_id}&t=${v.timestamp}s`,
          toHHMMSS(v.timestamp),
        ])
      )
    } catch {
      setError(t("Error fetching data. Please try again later."))
    }
  }
  return (
    <TooltipProvider>
      <div className="flex flex-col gap-4">
        <h2 className="text-2xl font-bold text-center flex items-center justify-center gap-2">
          {t("Funniest Timestamps")}
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-4 w-4 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-sm text-center">
              {t("Determined using the highest concentration of humerous reactions by chat for each stream.")}
              <br /><br />
              {t("Each link opens YouTube in a new window.")}
              <br /><br />
              {t("CSV Download contains timestamps in HH:MM:SS format for use with yt-dlp, ffmpeg, etc. for clipping.")}
            </TooltipContent>
          </Tooltip>
        </h2>
        <div className="flex flex-wrap items-end justify-center gap-4">
          <div className="space-y-1.5">
            <Label>{t("Channel:")}</Label>
            <Select value={channel} onValueChange={setChannel}>
              <SelectTrigger className="w-[220px]">
                <SelectValue placeholder={t("Select Channel")} />
              </SelectTrigger>
              <SelectContent>
                {channels.map((c) => (
                  <SelectItem key={c} value={c}>{c}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label>{t("Month:")}</Label>
            <MonthPicker value={month} onChange={setMonth} locale={i18n.language} className="w-[200px]" />
          </div>
          <Button onClick={fetchData} disabled={loading}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {t("Fetch Funniest Moments")}
          </Button>
          <Button variant="secondary" onClick={handleCSV}>
            {t("Download CSV")}
          </Button>
        </div>
        {error && <p className="text-center text-destructive text-sm">{error}</p>}
        <div className="rounded-lg bg-card p-4 min-h-[300px] max-h-[65vh] overflow-y-auto">
          {loading && (
            <div className="flex justify-center py-10">
              <Loader2 className="h-8 w-8 animate-spin text-primary" />
            </div>
          )}
          {!loading && rows && (
            <div
              className="grid gap-4 justify-center"
              style={{ gridTemplateColumns: "repeat(auto-fill, minmax(160px, 160px))" }}
            >
              {rows.map((v, i) => (
                <div key={i} className="text-center bg-secondary rounded-lg p-2 shadow hover:scale-105 transition-transform">
                  <img
                    src={`https://img.youtube.com/vi/${v.video_id}/hqdefault.jpg`}
                    alt={v.title}
                    className="w-[140px] h-[90px] object-cover rounded-md cursor-pointer mx-auto"
                    onClick={() =>
                      window.open(`https://www.youtube.com/watch?v=${v.video_id}&t=${v.timestamp}s`, "_blank")
                    }
                  />
                  <h6 className="text-sm mt-2 truncate" title={v.title}>{v.title}</h6>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </TooltipProvider>
  )
}