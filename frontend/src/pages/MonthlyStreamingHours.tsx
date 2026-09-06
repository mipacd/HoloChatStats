import { useCallback, useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { Area, CartesianGrid, ComposedChart, Line, XAxis, YAxis } from "recharts"
import {
  ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, type ChartConfig,
} from "@/components/ui/chart"
import { Label } from "@/components/ui/label"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { ChartShell } from "@/components/charts/ChartShell"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
interface ApiRow {
  month: string
  total_streaming_hours: number
  is_forecast: boolean
  confidence_low?: number
  confidence_high?: number
}
interface Row {
  month: string
  historical: number | null
  forecast: number | null
  band: [number, number] | null
}
const monthToNumber = (m: string) => {
  const [y, mo] = m.split("-").map(Number)
  return y * 12 + mo
}
const numberToMonth = (n: number) => {
  const year = Math.floor((n - 1) / 12)
  const month = ((n - 1) % 12) + 1
  return `${year}-${String(month).padStart(2, "0")}`
}
const currentMonthNumber = () => {
  const now = new Date()
  return now.getFullYear() * 12 + (now.getMonth() + 1)
}
function buildRows(data: ApiRow[]): Row[] {
  const historical = data.filter((d) => !d.is_forecast)
  const forecast = data.filter((d) => d.is_forecast)
  if (historical.length === 0) return []
  let showForecast = false
  if (forecast.length > 0) {
    const sorted = [...historical].sort((a, b) => monthToNumber(a.month) - monthToNumber(b.month))
    const last = sorted[sorted.length - 1]
    if (
      monthToNumber(last.month) === currentMonthNumber() - 1 &&
      last.total_streaming_hours > 0
    ) {
      showForecast = true
    }
  }
  const histMap = new Map(historical.map((d) => [d.month, d.total_streaming_hours]))
  const foreMap = new Map(
    forecast.map((d) => [d.month, { value: d.total_streaming_hours, low: d.confidence_low ?? null, high: d.confidence_high ?? null }])
  )
  const allMonths = showForecast
    ? [...historical.map((d) => d.month), ...forecast.map((d) => d.month)]
    : historical.map((d) => d.month)
  const nums = allMonths.map(monthToNumber)
  const min = Math.min(...nums)
  const max = Math.max(...nums)
  const rows: Row[] = []
  let lastHistValue: number | null = null
  let lastHistIndex = -1
  for (let n = min; n <= max; n++) {
    const m = numberToMonth(n)
    const hist = histMap.get(m)
    const fore = foreMap.get(m)
    if (!showForecast && fore && hist === undefined) continue
    if (hist !== undefined) {
      rows.push({ month: m, historical: hist, forecast: null, band: null })
      lastHistValue = hist
      lastHistIndex = rows.length - 1
    } else if (fore && showForecast) {
      rows.push({
        month: m,
        historical: null,
        forecast: fore.value,
        band: fore.low != null && fore.high != null ? [fore.low, fore.high] : null,
      })
    } else {
      rows.push({ month: m, historical: 0, forecast: null, band: null })
      lastHistValue = 0
      lastHistIndex = rows.length - 1
    }
  }
  // Connect last historical point into the forecast line/band
  if (showForecast && lastHistIndex >= 0 && lastHistValue != null && lastHistValue > 0) {
    rows[lastHistIndex].forecast = lastHistValue
    rows[lastHistIndex].band = [lastHistValue, lastHistValue]
  }
  return rows
}
export default function MonthlyStreamingHours() {
  const { t } = useTranslation()
  const [channels, setChannels] = useState<string[]>([])
  const [channel, setChannel] = useState("")
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    api.get("/get_channel_names").then((res) => setChannels(res.data || []))
  }, [])
  const fetchData = useCallback(async () => {
    if (!channel) return
    setLoading(true)
    try {
      const res = await api.get("/get_monthly_streaming_hours", { params: { channel } })
      setRows(buildRows(res.data || []))
    } catch {
      setRows([])
    } finally {
      setLoading(false)
    }
  }, [channel])
  useEffect(() => {
    fetchData()
  }, [fetchData])
  useEffect(() => {
    registerEriContext(() => ({
      page: "monthly_streaming_hours",
      endpoint: "/api/get_monthly_streaming_hours",
      parameters: { channel },
      description: `Viewing monthly streaming hours for ${channel}`,
    }))
    return () => registerEriContext(null)
  }, [channel])
  const config = {
    historical: { label: t("Streaming Hours"), color: "rgba(75,192,192,1)" },
    forecast: { label: t("Forecast"), color: "rgba(255,159,64,1)" },
  } satisfies ChartConfig
  const MonthlyTooltip = ({ active, label }: { active?: boolean; label?: string }) => {
    if (!active || !label) return null
    const row = rows.find((r) => r.month === label)
    if (!row) return null
    const lines: string[] = []
    if (row.historical != null) lines.push(`${t("Streaming Hours")}: ${row.historical.toFixed(2)} hours`)
    if (row.forecast != null && row.historical == null) {
      lines.push(`${t("Forecast")}: ${row.forecast.toFixed(2)} hours`)
      if (row.band) lines.push(`${t("P25-P75 Range")}: ${row.band[0].toFixed(2)} - ${row.band[1].toFixed(2)} hours`)
    }
    if (!lines.length) return null
    return (
      <div className="rounded-lg border border-border bg-popover px-3 py-2 text-xs shadow-xl">
        <p className="font-medium mb-1">{label}</p>
        {lines.map((l, i) => (
          <div key={i}>{l}</div>
        ))}
      </div>
    )
  }
  const chartMinWidth = useMemo(() => rows.length * 50, [rows])
  return (
    <ChartShell
      title={t("Monthly Streaming Hours")}
      infoText={t("Only counts public archived streams from YouTube.")}
      loading={loading}
      hasData={rows.length > 0}
      emptyText={channel ? t("No data available.") : t("Please select a channel.")}
      chartMinWidth={chartMinWidth}
      png={{ title: `Monthly Streaming Hours - ${channel} - holochatstats.info`, filename: "monthly_streaming_hours.png" }}
      controls={
        <div className="space-y-1.5">
          <Label>{t("Channel:")}</Label>
          <Select value={channel} onValueChange={setChannel}>
            <SelectTrigger className="w-[260px]">
              <SelectValue placeholder={t("Select Channel")} />
            </SelectTrigger>
            <SelectContent>
              {channels.map((c) => (
                <SelectItem key={c} value={c}>{c}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      }
    >
      <ChartContainer config={config} className="h-full w-full !aspect-auto">
        <ComposedChart data={rows} margin={{ top: 20, right: 30, left: 10, bottom: 50 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="month" tick={{ fill: "white", fontSize: 11 }} angle={-45} textAnchor="end" height={70} />
          <YAxis
            tick={{ fill: "white" }}
            domain={[0, (dataMax: number) => Math.ceil(dataMax * 1.05)]}
            label={{ value: t("Streaming Hours"), angle: -90, position: "insideLeft", fill: "white" }}
          />
          <ChartTooltip content={MonthlyTooltip as any} />
          <ChartLegend content={<ChartLegendContent />} />
          <Area
            dataKey="band"
            stroke="rgba(255,206,86,0.4)"
            fill="rgba(255,206,86,0.2)"
            strokeWidth={1}
            strokeDasharray="3 3"
            connectNulls={false}
            legendType="none"
            isAnimationActive={false}
          />
          <Line
            type="linear"
            dataKey="historical"
            name="historical"
            stroke="var(--color-historical)"
            strokeWidth={2}
            dot={{ r: 3 }}
            connectNulls={false}
          />
          <Line
            type="linear"
            dataKey="forecast"
            name="forecast"
            stroke="var(--color-forecast)"
            strokeWidth={2}
            strokeDasharray="10 5"
            dot={{ r: 4 }}
            connectNulls
          />
        </ComposedChart>
      </ChartContainer>
    </ChartShell>
  )
}