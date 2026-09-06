import { useCallback, useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Bar, BarChart, CartesianGrid, Cell, ReferenceLine, XAxis, YAxis } from "recharts"
import {
  ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig,
} from "@/components/ui/chart"
import { Label } from "@/components/ui/label"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { MonthPicker } from "@/components/ui/month-picker"
import { ChartShell } from "@/components/charts/ChartShell"
import { api } from "@/lib/api"
import { downloadCSV } from "@/lib/chart-export"
import { registerEriContext } from "@/components/eri/eri-context"
interface Row {
  channel: string
  change: number
}
const POS = "rgba(75,192,192,0.7)"
const NEG = "rgba(255,99,132,0.7)"
const lastMonth = () => {
  const d = new Date()
  d.setMonth(d.getMonth() - 1)
  return d.toISOString().slice(0, 7)
}
export default function StreamingHoursDiff() {
  const { t, i18n } = useTranslation()
  const [group, setGroup] = useState("all")
  const [month, setMonth] = useState(lastMonth())
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  const groupParam = group === "all" ? undefined : group
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.get("/get_group_streaming_hours_diff", {
        params: { month, ...(groupParam ? { group: groupParam } : {}) },
      })
      if (!res.data?.success) {
        setRows([])
        return
      }
      setRows((res.data.data || []).map((d: any) => ({ channel: d.channel, change: d.change })))
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
      page: "streaming_hours_diff",
      endpoint: "/api/get_group_streaming_hours_diff",
      parameters: { month, group: group === "all" ? "All" : group },
      description: `Viewing streaming hour changes for ${group === "all" ? "All" : group} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [group, month])
  const config = {
    change: { label: t("Change in Hours"), color: POS },
  } satisfies ChartConfig
  const handleCSV = () =>
    downloadCSV("streaming_hours_diff.csv", ["Channel", "HourChange"], rows.map((r) => [r.channel, r.change]))
  return (
    <ChartShell
      title={t("Streaming Hour Change Since Previous Month")}
      infoText={t("Archived, non-member streams from YouTube only.")}
      loading={loading}
      hasData={rows.length > 0}
      chartMinWidth={rows.length * 42}
      onDownloadCSV={handleCSV}
      png={{
        title: `Streaming Hour Change Since Previous Month - ${group === "all" ? "All" : group} - ${month} - holochatstats.info`,
        filename: "streaming_hour_diff.png",
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
          <YAxis tick={{ fill: "white" }} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Bar dataKey="change">
            {rows.map((r, i) => (
              <Cell key={i} fill={r.change >= 0 ? POS : NEG} />
            ))}
          </Bar>
        </BarChart>
      </ChartContainer>
    </ChartShell>
  )
}