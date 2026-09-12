import { useCallback, useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import {
  ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig,
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
  message_rate: number
}
export default function MessageTypes() {
  const { t } = useTranslation()
  const [channels, setChannels] = useState<string[]>([])
  const [channel, setChannel] = useState("")
  const [language, setLanguage] = useState("EN")
  const [rows, setRows] = useState<Row[]>([])
  const [loading, setLoading] = useState(false)
  useEffect(() => {
    api.get("/get_channel_names").then((res) => setChannels(res.data || []))
  }, [])
  const fetchData = useCallback(async () => {
    if (!channel || !language) return
    setLoading(true)
    try {
      const res = await api.get("/get_message_type_percents", {
        params: { channel, language },
      })
      setRows(res.data || [])
    } catch {
      setRows([])
    } finally {
      setLoading(false)
    }
  }, [channel, language])
  useEffect(() => {
    fetchData()
  }, [fetchData])
  useEffect(() => {
    registerEriContext(() => ({
      page: "message_types",
      endpoint: "/api/get_message_type_percents",
      parameters: { channel, language },
      description: `Viewing message percentages and rates by language for ${channel} in ${language}`,
    }))
    return () => registerEriContext(null)
  }, [channel, language])
  const config = {
    percent: { label: t("Percentage"), color: "rgba(75,192,192,1)" },
    message_rate: { label: t("Rate (messages/min)"), color: "rgba(255,99,132,1)" },
  } satisfies ChartConfig
  return (
    <ChartShell
      title={t("Language Percentages / Rates")}
      infoText={t("Emote messages are excluded from the total count. Total duration is computed using only streams with available chat logs.")}
      loading={loading}
      hasData={rows.length > 0}
      emptyText={channel ? t("No data available for the selected channel and language.") : t("Please select a channel.")}
      chartMinWidth={rows.length * 50}
      png={{
        title: `Language Percentages / Rates - ${channel} (${language}) - holochatstats.info`,
        filename: "message_types.png",
      }}
      controls={
        <>
          <div className="space-y-1.5">
            <Label>{t("Channel:")}</Label>
            <Select value={channel} onValueChange={setChannel}>
              <SelectTrigger className="w-[240px]">
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
            <Label>{t("Language:")}</Label>
            <Select value={language} onValueChange={setLanguage}>
              <SelectTrigger className="w-[240px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="EN">{t("English (or none of the others)")}</SelectItem>
                <SelectItem value="JP">{t("Japanese")}</SelectItem>
                <SelectItem value="KR">{t("Korean")}</SelectItem>
                <SelectItem value="RU">{t("Russian")}</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </>
      }
    >
      <ChartContainer config={config} className="h-full w-full !aspect-auto">
        <LineChart data={rows} margin={{ top: 20, right: 30, left: 10, bottom: 50 }}>
          <CartesianGrid stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="month" tick={{ fill: "white", fontSize: 11 }} angle={-45} textAnchor="end" height={70} />
          <YAxis
            yAxisId="left"
            tick={{ fill: "white" }}
            label={{ value: t("Percentage"), angle: -90, position: "insideLeft", fill: "white" }}
          />
          <YAxis
            yAxisId="right"
            orientation="right"
            tick={{ fill: "white" }}
            label={{ value: t("Rate (messages/min)"), angle: 90, position: "insideRight", fill: "white" }}
          />
          <ChartTooltip
            content={
              <ChartTooltipContent
                formatter={(value, _name, item) =>
                  `${value}${item.dataKey === "percent" ? "%" : " messages/min"}`
                }
              />
            }
          />
          <ChartLegend content={<ChartLegendContent />} />
          <Line yAxisId="left" type="linear" dataKey="percent" stroke="var(--color-percent)" strokeWidth={2} dot={false} />
          <Line yAxisId="right" type="linear" dataKey="message_rate" stroke="var(--color-message_rate)" strokeWidth={2} dot={false} />
        </LineChart>
      </ChartContainer>
    </ChartShell>
  )
}
