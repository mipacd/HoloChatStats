import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import {
  ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig,
} from "@/components/ui/chart"
import { Label } from "@/components/ui/label"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { ChartShell } from "@/components/charts/ChartShell"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
interface Row {
  month: string
  percent: number
}
export default function ExclusiveChat() {
  const { t } = useTranslation()
  const [channels, setChannels] = useState<string[]>([])
  const [channel, setChannel] = useState("")
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    api.get("/get_channel_names").then((res) => setChannels(res.data || []))
  }, [])
  useEffect(() => {
    if (!channel) return
    setLoading(true)
    api
      .get("/get_exclusive_chat_users", { params: { channel } })
      .then((res) => setRows(res.data || []))
      .catch(() => setRows([]))
      .finally(() => setLoading(false))
  }, [channel])
  useEffect(() => {
    registerEriContext(() => ({
      page: "exclusive_chat",
      endpoint: "/api/get_exclusive_chat_users",
      parameters: { channel },
      description: `Viewing exclusive chat user rates for ${channel}`,
    }))
    return () => registerEriContext(null)
  }, [channel])
  const config = {
    percent: { label: t("Exclusive Users (%)"), color: "rgba(75,192,192,1)" },
  } satisfies ChartConfig
  return (
    <ChartShell
      title={t("Exclusive Chat Users")}
      infoText={t("Percentage of chat users exclusive to this channel each month, within the same group (e.g. Hololive, Indie).")}
      loading={loading}
      hasData={rows.length > 0}
      emptyText={channel ? t("No data available for the selected channel.") : t("Please select a channel.")}
      chartMinWidth={rows.length * 50}
      png={{ title: `Exclusive Chat Users - ${channel} - holochatstats.info`, filename: "exclusive_chat_users.png" }}
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
        <LineChart data={rows} margin={{ top: 30, right: 30, left: 10, bottom: 50 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="month" tick={{ fill: "white", fontSize: 11 }} angle={-45} textAnchor="end" height={70} />
          <YAxis
            tick={{ fill: "white" }}
            label={{ value: t("Exclusive Users (%)"), angle: -90, position: "insideLeft", fill: "white" }}
          />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Line type="linear" dataKey="percent" stroke="var(--color-percent)" strokeWidth={2} dot={false} />
        </LineChart>
      </ChartContainer>
    </ChartShell>
  )
}