import { useCallback, useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { Bar, BarChart, CartesianGrid, ReferenceLine, XAxis, YAxis } from "recharts"
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
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
  users_gained: number
  users_lost: number
}
const lastMonth = () => {
  const d = new Date()
  d.setMonth(d.getMonth() - 1)
  return d.toISOString().slice(0, 7)
}
export default function UserChange() {
  const { t, i18n } = useTranslation()
  const [group, setGroup] = useState("Hololive")
  const [month, setMonth] = useState(lastMonth())
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  const groupParam = group === "all" ? undefined : group
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.get("/get_user_changes", {
        params: { month, ...(groupParam ? { group: groupParam } : {}) },
      })
      const data: Row[] = res.data || []
      data.sort((a, b) => b.users_gained - b.users_lost - (a.users_gained - a.users_lost))
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
      page: "user_change",
      endpoint: "/api/get_user_changes",
      parameters: { month, group: group === "all" ? "All" : group },
      description: `Viewing active user gains/losses for ${group === "all" ? "All" : group} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [group, month])
  const chartData = useMemo(
    () => rows.map((d) => ({ channel: d.channel, gained: d.users_gained, lost: -d.users_lost })),
    [rows]
  )
  const config = {
    gained: { label: t("Users Gained"), color: "rgba(75,192,192,0.8)" },
    lost: { label: t("Users Lost"), color: "rgba(255,99,132,0.8)" },
  } satisfies ChartConfig
  const handleCSV = () =>
    downloadCSV(
      "user_changes.csv",
      ["Channel", "Users Gained", "Users Lost", "Net Change"],
      rows.map((d) => [d.channel, d.users_gained, d.users_lost, d.users_gained - d.users_lost])
    )
  return (
    <ChartShell
      title={t("Active User Gains / Losses")}
      infoText={t("Determined by users who met a 5 message threshold in one month but not in the other. Channels that didn't stream in either month are not included.")}
      loading={loading}
      hasData={chartData.length > 0}
      chartMinWidth={chartData.length * 42}
      onDownloadCSV={handleCSV}
      png={{
        title: `Active User Gains/Losses - ${group === "all" ? "All" : group} - ${month} - holochatstats.info`,
        filename: "user_change.png",
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
          <Bar dataKey="gained" stackId="a" fill="var(--color-gained)" />
          <Bar dataKey="lost" stackId="a" fill="var(--color-lost)" />
        </BarChart>
      </ChartContainer>
    </ChartShell>
  )
}