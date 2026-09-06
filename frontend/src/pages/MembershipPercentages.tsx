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
type RawRow = [string, number, number, number]
interface Row {
  channel: string
  percent: number
}
const lastMonth = () => {
  const d = new Date()
  d.setMonth(d.getMonth() - 1)
  return d.toISOString().slice(0, 7)
}
export default function MembershipPercentages() {
  const { t, i18n } = useTranslation()
  const [group, setGroup] = useState("Hololive")
  const [month, setMonth] = useState(lastMonth())
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.get("/get_group_membership_data", {
        params: { month, channel_group: group },
      })
      const data: RawRow[] = res.data || []
      const processed = data
        .filter((d) => d[1] === -1)
        .map((d) => ({ channel: d[0], percent: 100 - parseFloat(String(d[3])) }))
        .sort((a, b) => b.percent - a.percent)
      setRows(processed)
    } catch {
      setRows([])
    } finally {
      setLoading(false)
    }
  }, [group, month])
  useEffect(() => {
    fetchData()
  }, [fetchData])
  useEffect(() => {
    registerEriContext(() => ({
      page: "membership_percentages",
      endpoint: "/api/get_group_membership_data",
      parameters: { month, channel_group: group },
      description: `Viewing membership percentages for ${group} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [group, month])
  const config = {
    percent: { label: t("Membership Percentage"), color: "rgba(75,192,192,0.8)" },
  } satisfies ChartConfig
  const handleCSV = () =>
    downloadCSV("membership_percentages.csv", ["Channel", "Percentage"], rows.map((r) => [r.channel, r.percent]))
  return (
    <ChartShell
      title={t("Membership Percentages")}
      infoText={t("Only counts members that participated in chat.")}
      loading={loading}
      hasData={rows.length > 0}
      chartMinWidth={rows.length * 42}
      onDownloadCSV={handleCSV}
      png={{
        title: `Membership Percentages - ${group} - ${month} - holochatstats.info`,
        filename: "membership_percents.png",
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
        <BarChart data={rows} margin={{ top: 30, right: 20, left: 0, bottom: 80 }}>
          <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="channel" interval={0} angle={-45} textAnchor="end" height={90} tick={{ fill: "white", fontSize: 11 }} />
          <YAxis
            tick={{ fill: "white" }}
            domain={[0, (dataMax: number) => Math.ceil(dataMax * 1.05)]}
            />
          <ChartTooltip
            content={
              <ChartTooltipContent formatter={(value) => `${Number(value).toFixed(2)}%`} />
            }
          />
          <Bar dataKey="percent" fill="var(--color-percent)" radius={[2, 2, 0, 0]} />
        </BarChart>
      </ChartContainer>
    </ChartShell>
  )
}