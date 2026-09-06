import { useEffect, useState } from "react"
import { useSearchParams } from "react-router-dom"
import { useTranslation } from "react-i18next"
import { Loader2 } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Button } from "@/components/ui/button"
import { Table, TableBody, TableCell, TableRow } from "@/components/ui/table"
import { MonthPicker } from "@/components/ui/month-picker"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
type Mode = "users" | "members"
const MODE_CONFIG: Record<
  Mode,
  {
    endpoint: string
    noResults: string
    rows: { key: string; suffix: string; template: string }[]
  }
> = {
  users: {
    endpoint: "/get_common_users",
    noResults: "No common users found for the selected criteria.",
    rows: [
      {
        key: "percent_A_to_B_users",
        suffix: "%",
        template: "Percent of {{a}}'s chat in {{ma}} that participated in {{b}}'s chat in {{mb}}:",
      },
      {
        key: "percent_B_to_A_users",
        suffix: "%",
        template: "Percent of {{b}}'s chat in {{mb}} that participated in {{a}}'s chat in {{ma}}:",
      },
      { key: "num_common_users", suffix: "", template: "Number of common users:" },
    ],
  },
  members: {
    endpoint: "/get_common_members",
    noResults: "No common members found for the selected criteria.",
    rows: [
      {
        key: "percent_A_to_B_members",
        suffix: "%",
        template: "Percent of {{a}}'s members in {{ma}} that are also members of {{b}} in {{mb}}:",
      },
      {
        key: "percent_B_to_A_members",
        suffix: "%",
        template: "Percent of {{b}}'s members in {{mb}} that are also members of {{a}} in {{ma}}:",
      },
      { key: "num_common_members", suffix: "", template: "Number of common members:" },
    ],
  },
}
function formatMonthYear(value: string, locale: string) {
  if (!value) return value
  const [y, m] = value.split("-").map(Number)
  if (!y || !m) return value
  return new Intl.DateTimeFormat(locale, { year: "numeric", month: "long" }).format(
    new Date(y, m - 1, 1)
  )
}
export default function CommonStats() {
  const { t, i18n } = useTranslation()
  const [searchParams] = useSearchParams()
  const [mode, setMode] = useState<Mode>(
    searchParams.get("mode") === "members" ? "members" : "users"
  )
  const [channels, setChannels] = useState<string[]>([])
  const [channelA, setChannelA] = useState("")
  const [channelB, setChannelB] = useState("")
  const [monthA, setMonthA] = useState("")
  const [monthB, setMonthB] = useState("")
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<Record<string, number> | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.get("/get_channel_names").then((res) => setChannels(res.data || []))
  }, [])
  // Feed current selections to the Eri widget as page context
  useEffect(() => {
    registerEriContext(() => ({
      page: mode === "users" ? "common_users" : "common_members",
      endpoint: MODE_CONFIG[mode].endpoint,
      parameters: { channel_a: channelA, month_a: monthA, channel_b: channelB, month_b: monthB },
      description: `Viewing common ${mode} data between ${channelA} (${monthA}) and ${channelB} (${monthB})`,
    }))
    return () => registerEriContext(null)
  }, [mode, channelA, channelB, monthA, monthB])
  const config = MODE_CONFIG[mode]
  const handleCalculate = async () => {
    if (!channelA || !monthA || !channelB || !monthB) {
      setError(t("Please fill in all fields."))
      return
    }
    setError(null)
    setLoading(true)
    setResults(null)
    try {
      const res = await api.get(config.endpoint, {
        params: { channel_a: channelA, month_a: monthA, channel_b: channelB, month_b: monthB },
      })
      if (!res.data || Object.keys(res.data).length === 0) {
        setError(t(config.noResults))
      } else {
        setResults(res.data)
      }
    } catch {
      setError(t("Error fetching data. Please try again later."))
    } finally {
      setLoading(false)
    }
  }
  const title =
    mode === "users"
      ? t("Common Users by Channel / Month")
      : t("Common Members by Channel / Month")
  return (
    <div className="flex flex-col gap-6 max-w-3xl mx-auto w-full">
      <h2 className="text-2xl font-bold text-center">{title}</h2>
      <div className="flex justify-center">
        <Tabs
          value={mode}
          onValueChange={(v) => {
            setMode(v as Mode)
            setResults(null)
            setError(null)
          }}
        >
          <TabsList>
            <TabsTrigger value="users">{t("Common Users")}</TabsTrigger>
            <TabsTrigger value="members">{t("Common Members")}</TabsTrigger>
          </TabsList>
        </Tabs>
      </div>
      <Card>
        <CardContent className="pt-6 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">{t("Channel A:")}</label>
              <Select value={channelA} onValueChange={setChannelA}>
                <SelectTrigger className="max-w-[300px]">
                  <SelectValue placeholder={t("Select Channel")} />
                </SelectTrigger>
                <SelectContent>
                  {channels.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">{t("Month for A:")}</label>
              <MonthPicker value={monthA} onChange={setMonthA} placeholder={t("Select month")} locale={i18n.language} />
            </div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <label className="text-sm font-medium">{t("Channel B:")}</label>
              <Select value={channelB} onValueChange={setChannelB}>
                <SelectTrigger className="max-w-[300px]">
                  <SelectValue placeholder={t("Select Channel")} />
                </SelectTrigger>
                <SelectContent>
                  {channels.map((c) => (
                    <SelectItem key={c} value={c}>
                      {c}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-1.5">
              <label className="text-sm font-medium">{t("Month for B:")}</label>
              <MonthPicker value={monthB} onChange={setMonthB} placeholder={t("Select month")} locale={i18n.language} />
            </div>
          </div>
          <div className="text-center pt-2">
            <Button onClick={handleCalculate} disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t("Calculate")}
            </Button>
          </div>
          {error && <p className="text-center text-destructive text-sm">{error}</p>}
        </CardContent>
      </Card>
      {loading && (
        <div className="flex justify-center py-6">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      )}
      {results && !loading && (
        <Card>
          <CardContent className="pt-6">
            <Table>
              <TableBody>
                {config.rows.map((row) => {
                  const label = t(row.template, {
                    defaultValue: row.template,
                    a: channelA,
                    ma: formatMonthYear(monthA, i18n.language),
                    b: channelB,
                    mb: formatMonthYear(monthB, i18n.language),
                  })
                  const raw = results[row.key]
                  const value = raw !== undefined && raw !== null ? `${raw}${row.suffix}` : "N/A"
                  return (
                    <TableRow key={row.key}>
                      <TableCell className="font-medium w-2/3">{label}</TableCell>
                      <TableCell>{value}</TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}