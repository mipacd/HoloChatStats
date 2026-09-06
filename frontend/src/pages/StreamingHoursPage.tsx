import { useCallback, useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"
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
  hours: number
}
const lastMonth = () => {
  const d = new Date()
  d.setMonth(d.getMonth() - 1)
  return d.toISOString().slice(0, 7)
}
interface Props {
  pageId: string
  endpoint: string
  title: string
  infoText: string
  pngTitlePrefix: string
  csvFilename: string
  pngFilename: string
  defaultGroup?: string
}
export function StreamingHoursPage({
  pageId,
  endpoint,
  title,
  infoText,
  pngTitlePrefix,
  csvFilename,
  pngFilename,
  defaultGroup = "all",
}: Props) {
  const { t, i18n } = useTranslation()
  const [group, setGroup] = useState(defaultGroup)
  const [month, setMonth] = useState(lastMonth())
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  const groupParam = group === "all" ? undefined : group
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.get(endpoint, {
        params: { month, ...(groupParam ? { group: groupParam } : {}) },
      })
      if (!res.data?.success) {
        setRows([])
        return
      }
      setRows((res.data.data || []).map((d: any) => ({ channel: d.channel, hours: d.hours })))
    } catch {
      setRows([])
    } finally {
      setLoading(false)
    }
  }, [endpoint, groupParam, month])
  useEffect(() => {
    fetchData()
  }, [fetchData])
  useEffect(() => {
    registerEriContext(() => ({
      page: pageId,
      endpoint: `/api${endpoint}`,
      parameters: { month, group: group === "all" ? "All" : group },
      description: `Viewing ${pageId} for ${group === "all" ? "All" : group} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [pageId, endpoint, group, month])
  const config = {
    hours: { label: t("Hours"), color: "rgba(75,192,192,0.8)" },
  } satisfies ChartConfig
  const handleCSV = () =>
    downloadCSV(csvFilename, ["Channel", "Hours"], rows.map((r) => [r.channel, r.hours]))
  return (
    <ChartShell
      title={title}
      infoText={infoText}
      loading={loading}
      hasData={rows.length > 0}
      chartMinWidth={rows.length * 42}
      onDownloadCSV={handleCSV}
      png={{
        title: `${pngTitlePrefix} - ${group === "all" ? "All" : group} - ${month} - holochatstats.info`,
        filename: pngFilename,
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
          <Bar dataKey="hours" fill="var(--color-hours)" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ChartContainer>
    </ChartShell>
  )
}