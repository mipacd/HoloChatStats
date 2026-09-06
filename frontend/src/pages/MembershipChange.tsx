import { useCallback, useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { Bar, BarChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts"
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
import { registerEriContext } from "@/components/eri/eri-context"
interface Row {
  channel_name: string
  gains_count: number
  losses_count: number
  differential: number
}
const lastMonth = () => {
  const d = new Date()
  d.setMonth(d.getMonth() - 1)
  return d.toISOString().slice(0, 7)
}
export default function MembershipChange() {
  const { t, i18n } = useTranslation()
  const [group, setGroup] = useState("Hololive")
  const [month, setMonth] = useState(lastMonth())
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  const groupParam = group === "all" ? undefined : group
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.get("/get_group_membership_changes", {
        params: { month, ...(groupParam ? { channel_group: groupParam } : {}) },
      })
      const data: Row[] = res.data || []
      data.sort((a, b) => b.differential - a.differential)
      setRows(data)
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
      page: "membership_changes",
      endpoint: "/api/get_group_membership_changes",
      parameters: { month, channel_group: group === "all" ? "" : group },
      description: `Viewing membership changes for ${group === "all" ? "all groups" : group} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [group, month])
  const chartData = useMemo(
    () => rows.map((d) => ({ channel: d.channel_name, gains: d.gains_count, losses: -d.losses_count })),
    [rows]
  )
  const config = {
    gains: { label: t("Gains"), color: "rgba(75,192,192,0.8)" },
    losses: { label: t("Losses"), color: "rgba(255,99,132,0.8)" },
  } satisfies ChartConfig
  const handleCSV = () =>
    downloadCSV(
      "membership_changes.csv",
      ["Channel", "Gains", "Losses", "Differential"],
      rows.map((d) => [d.channel_name, d.gains_count, d.losses_count, d.differential])
    )
  return (
    <ChartShell
      title={t("Membership Gain / Loss")}
      infoText={t("Determined using the last recorded message for each user between the specified month and the previous month.")}
      loading={loading}
      hasData={chartData.length > 0}
      chartMinWidth={chartData.length * 42}
      onDownloadCSV={handleCSV}
      png={{
        title: `Membership Gain / Loss - ${group === "all" ? "" : group} - ${month} - holochatstats.info`,
        filename: "membership_changes.png",
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
        <BarChart data={chartData} stackOffset="sign" margin={{ top: 30, right: 20, left: 0, bottom: 70 }}>
          <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="channel" interval={0} angle={-45} textAnchor="end" height={80} tick={{ fill: "white", fontSize: 11 }} />
          <YAxis tick={{ fill: "white" }} />
          <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <ChartLegend content={<ChartLegendContent />} />
          <Bar dataKey="gains" stackId="a" fill="var(--color-gains)" />
          <Bar dataKey="losses" stackId="a" fill="var(--color-losses)" />
        </BarChart>
      </ChartContainer>
    </ChartShell>
  )
}