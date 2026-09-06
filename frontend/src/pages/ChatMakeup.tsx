import { useCallback, useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
import {
  ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig,
} from "@/components/ui/chart"
import { Label } from "@/components/ui/label"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { MonthPicker } from "@/components/ui/month-picker"
import { ChartShell } from "@/components/charts/ChartShell"
import { api } from "@/lib/api"
import { downloadCSV } from "@/lib/chart-export"
import { TABLEAU10 } from "@/lib/chart-colors"
import { registerEriContext } from "@/components/eri/eri-context"
interface RawRow {
  channel_name: string
  es_en_id_rate_per_minute: number
  jp_rate_per_minute: number
  kr_rate_per_minute: number
  ru_rate_per_minute: number
  emoji_rate_per_minute: number
}
interface Row {
  channel: string
  en: number
  jp: number
  kr: number
  ru: number
  emote: number
}
const lastMonth = () => {
  const d = new Date()
  d.setMonth(d.getMonth() - 1)
  return d.toISOString().slice(0, 7)
}
export default function ChatMakeup() {
  const { t, i18n } = useTranslation()
  const [group, setGroup] = useState("all")
  const [month, setMonth] = useState(lastMonth())
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  const groupParam = group === "all" ? undefined : group
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.get("/get_group_chat_makeup", {
        params: { month, ...(groupParam ? { group: groupParam } : {}) },
      })
      if (!res.data?.success) {
        setRows([])
        return
      }
      const data: RawRow[] = res.data.data || []
      const mapped: Row[] = data.map((d) => ({
        channel: d.channel_name,
        en: d.es_en_id_rate_per_minute,
        jp: d.jp_rate_per_minute,
        kr: d.kr_rate_per_minute,
        ru: d.ru_rate_per_minute,
        emote: d.emoji_rate_per_minute,
      }))
      mapped.sort(
        (a, b) =>
          b.en + b.jp + b.kr + b.ru + b.emote - (a.en + a.jp + a.kr + a.ru + a.emote)
      )
      setRows(mapped)
    } catch {
      setRows([])
    } finally {
      setLoading(false)
    }
  }, [groupParam, month])
  useEffect(() => {
    fetchData()
  }, [fetchData])
  useEffect(() => {
    registerEriContext(() => ({
      page: "chat_makeup",
      endpoint: "/api/get_group_chat_makeup",
      parameters: { month, group: group === "all" ? "" : group },
      description: `Viewing group chat makeup data for ${group === "all" ? "All" : group} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [group, month])
  const config = useMemo(
    () =>
      ({
        en: { label: "EN/ES/ID/etc.", color: TABLEAU10[0] },
        jp: { label: "JP", color: TABLEAU10[1] },
        kr: { label: "KR", color: TABLEAU10[2] },
        ru: { label: "RU", color: TABLEAU10[3] },
        emote: { label: "Emote", color: TABLEAU10[4] },
      }) satisfies ChartConfig,
    []
  )
  const handleCSV = () =>
    downloadCSV(
      "chat_makeup.csv",
      ["Channel", "EN/ES/ID/etc.", "JP", "KR", "RU", "Emote"],
      rows.map((r) => [r.channel, r.en, r.jp, r.kr, r.ru, r.emote])
    )
  return (
    <ChartShell
      title={t("Chat Makeup")}
      infoText={t("Rates per minute using character set detection. Emote category counts both Unicode and YouTube style emote-only messages.")}
      loading={loading}
      hasData={rows.length > 0}
      chartMinWidth={rows.length * 50}
      onDownloadCSV={handleCSV}
      png={{
        title: `Chat Makeup (avg. messages per min.) - ${group === "all" ? "All" : group} - ${month} - holochatstats.info`,
        filename: "chat_makeup.png",
      }}
      controls={
        <>
          <div className="space-y-1.5">
            <Label>{t("Channel Group:")}</Label>
            <Select value={group} onValueChange={setGroup}>
              <SelectTrigger className="w-[200px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">{t("All")}</SelectItem>
                <SelectItem value="Hololive">{t("Hololive")}</SelectItem>
                <SelectItem value="Indie">{t("Indie")}</SelectItem>
                <SelectItem value="PNGTubers">{t("PNGTubers")}</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-1.5">
            <Label>{t("Month:")}</Label>
            <MonthPicker value={month} onChange={setMonth} locale={i18n.language} className="w-[200px]" />
          </div>
        </>
      }
    >
      <ChartContainer config={config} className="h-full w-full !aspect-auto">
        <BarChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 80 }}>
          <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="channel" interval={0} angle={-45} textAnchor="end" height={90} tick={{ fill: "white", fontSize: 11 }} />
          <YAxis
            tick={{ fill: "white" }}
            domain={[0, (dataMax: number) => Math.ceil(dataMax * 1.05)]}
            />
          <ChartTooltip content={<ChartTooltipContent />} />
          <ChartLegend verticalAlign="top" content={<ChartLegendContent />} />
          <Bar dataKey="en" stackId="a" fill="var(--color-en)" />
          <Bar dataKey="jp" stackId="a" fill="var(--color-jp)" />
          <Bar dataKey="kr" stackId="a" fill="var(--color-kr)" />
          <Bar dataKey="ru" stackId="a" fill="var(--color-ru)" />
          <Bar dataKey="emote" stackId="a" fill="var(--color-emote)" />
        </BarChart>
      </ChartContainer>
    </ChartShell>
  )
}