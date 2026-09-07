import { useCallback, useEffect, useMemo, useState } from "react"
import { useTranslation } from "react-i18next"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { Label } from "@/components/ui/label"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { MonthPicker } from "@/components/ui/month-picker"
import { ChartShell } from "@/components/charts/ChartShell"
import { api } from "@/lib/api"
import { downloadCSV } from "@/lib/chart-export"
import { registerEriContext } from "@/components/eri/eri-context"
type RawRow = [string, number, number]
type Row = Record<string, number | string> & { channel: string; __total: number }
const COLOR_PALETTE = [
  "#4a6b91", "#8aa56f", "#b26e5f", "#9b7391", "#d1af75",
  "#6fa6b5", "#a54d55", "#6b8cad", "#b2b2b2", "#3c475a",
]
const NON_MEMBERS_COLOR = "#666666"
const UNKNOWN_COLOR = "#9370DB"
const NON_MEMBERS = "Non-Members"
const GIFT_ONLY = "Gift Only"
function formatMembershipLabel(rank: number) {
  if (rank === 0) return "New Members"
  if (rank < 12) return `${rank} Month${rank > 1 ? "s" : ""}`
  return `${Math.floor(rank / 12)} Year${rank >= 24 ? "s" : ""}`
}
function rankSortOrder(label: string) {
  if (label === "New Members") return 0
  if (label.includes("Month")) return parseInt(label.split(" ")[0], 10)
  if (label.includes("Year")) return parseInt(label.split(" ")[0], 10) * 12
  return Infinity
}
function processMembershipData(data: RawRow[]) {
  const grouped: Record<
    string,
    { total: number; counts: Record<string, number>; nonMembers: number; unknown: number }
  > = {}
  data.forEach(([channel, rank, count]) => {
    if (!grouped[channel]) grouped[channel] = { total: 0, counts: {}, nonMembers: 0, unknown: 0 }
    if (rank === -1) {
      grouped[channel].nonMembers = count
    } else if (rank === -2) {
      grouped[channel].unknown = count
      grouped[channel].total += count
    } else {
      const label = formatMembershipLabel(rank)
      grouped[channel].counts[label] = count
      grouped[channel].total += count
    }
  })
  const tierLabels = Array.from(
    new Set(Object.values(grouped).flatMap((c) => Object.keys(c.counts)))
  ).sort((a, b) => rankSortOrder(a) - rankSortOrder(b))
  const hasUnknown = Object.values(grouped).some((c) => c.unknown > 0)
  const rows: Row[] = Object.keys(grouped).map((channel) => {
    const row: Row = { channel, __total: grouped[channel].total }
    row[NON_MEMBERS] = grouped[channel].nonMembers || 0
    if (hasUnknown) row[GIFT_ONLY] = grouped[channel].unknown || 0
    tierLabels.forEach((label) => {
      row[label] = grouped[channel].counts[label] || 0
    })
    return row
  })
  const seriesKeys = [NON_MEMBERS, ...(hasUnknown ? [GIFT_ONLY] : []), ...tierLabels]
  return { rows, seriesKeys, tierLabels }
}
function colorFor(key: string, tierLabels: string[]) {
  if (key === NON_MEMBERS) return NON_MEMBERS_COLOR
  if (key === GIFT_ONLY) return UNKNOWN_COLOR
  const idx = tierLabels.indexOf(key)
  return COLOR_PALETTE[idx % COLOR_PALETTE.length]
}
const lastMonth = () => {
  const d = new Date()
  d.setMonth(d.getMonth() - 1)
  return d.toISOString().slice(0, 7)
}
export default function MembershipCounts() {
  const { t, i18n } = useTranslation()
  const [group, setGroup] = useState("Hololive")
  const [month, setMonth] = useState(lastMonth())
  const [rows, setRows] = useState<Row[]>([])
  const [seriesKeys, setSeriesKeys] = useState<string[]>([])
  const [tierLabels, setTierLabels] = useState<string[]>([])
  const [hidden, setHidden] = useState<Set<string>>(new Set([NON_MEMBERS]))
  const [loading, setLoading] = useState(false)
  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const res = await api.get("/get_group_membership_data", {
        params: { month, channel_group: group },
      })
      const processed = processMembershipData(res.data || [])
      setRows(processed.rows)
      setSeriesKeys(processed.seriesKeys)
      setTierLabels(processed.tierLabels)
    } catch {
      setRows([])
      setSeriesKeys([])
      setTierLabels([])
    } finally {
      setLoading(false)
    }
  }, [group, month])
  useEffect(() => {
    fetchData()
  }, [fetchData])
  useEffect(() => {
    registerEriContext(() => ({
      page: "group_membership",
      endpoint: "/api/get_group_membership_data",
      parameters: { month, channel_group: group },
      description: `Viewing group membership data for ${group} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [group, month])
  const nonMembersVisible = !hidden.has(NON_MEMBERS)
  const sortedRows = useMemo(
    () =>
        [...rows].sort((a, b) => {
        const totalA = seriesKeys.reduce((sum, k) => sum + (hidden.has(k) ? 0 : (a[k] as number)), 0)
        const totalB = seriesKeys.reduce((sum, k) => sum + (hidden.has(k) ? 0 : (b[k] as number)), 0)
        return totalB - totalA
        }),
    [rows, seriesKeys, hidden]
)
  const toggleKey = (key: string) =>
    setHidden((prev) => {
      const next = new Set(prev)
      next.has(key) ? next.delete(key) : next.add(key)
      return next
    })
  const CustomTooltip = ({ active, label }: { active?: boolean; label?: string }) => {
    if (!active || !label) return null
    const row = sortedRows.find((r) => r.channel === label)
    if (!row) return null
    const totalMembers = row.__total
    const totalChatters = totalMembers + (row[NON_MEMBERS] as number)
    const lines = seriesKeys
      .filter((k) => !hidden.has(k) && (row[k] as number) > 0)
      .map((k) => ({ key: k, value: row[k] as number, color: colorFor(k, tierLabels) }))
    return (
      <div className="rounded-lg border border-border bg-popover px-3 py-2 text-xs shadow-xl min-w-[11rem]">
        <p className="font-medium mb-1">{label}</p>
        {lines.map((l) => (
          <div key={l.key} className="flex justify-between gap-4">
            <span className="flex items-center gap-1.5 text-muted-foreground">
              <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: l.color }} />
              {l.key}
            </span>
            <span className="font-mono">{l.value}</span>
          </div>
        ))}
        <div className="mt-1 pt-1 border-t border-border/50 flex flex-col gap-0.5">
          <div className="flex justify-between gap-4">
            <span>{t("Total Members:")}</span>
            <span className="font-mono">{totalMembers}</span>
          </div>
          {nonMembersVisible && (
            <div className="flex justify-between gap-4">
              <span>{t("Total Chatters:")}</span>
              <span className="font-mono">{totalChatters}</span>
            </div>
          )}
        </div>
      </div>
    )
  }
  const handleCSV = () =>
    downloadCSV(
      "membership_counts.csv",
      ["Channel", ...seriesKeys],
      sortedRows.map((r) => [r.channel, ...seriesKeys.map((k) => r[k] as number)])
    )
  return (
    <ChartShell
      title={t("Membership Counts")}
      infoText={t("Based on chat log contents. Membership renewals/gifts are included from Nov. 2025. Membership durations are based on available badges for each channel.")}
      loading={loading}
      hasData={sortedRows.length > 0}
      chartMinWidth={sortedRows.length * 50}
      onDownloadCSV={handleCSV}
      png={{
        title: `Membership Counts - ${group} - ${month} - holochatstats.info`,
        filename: "membership_counts.png",
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
      extraHeader={
        seriesKeys.length > 0 && (
            <div className="flex flex-wrap justify-center gap-3 text-xs">
            {seriesKeys.map((key) => {
                const isHidden = hidden.has(key)
                return (
                <button
                    key={key}
                    type="button"
                    onClick={() => toggleKey(key)}
                    className="flex items-center gap-1.5"
                    style={{ opacity: isHidden ? 0.4 : 1 }}
                >
                    <span
                    className="inline-block w-3 h-3 rounded-sm"
                    style={{ backgroundColor: colorFor(key, tierLabels) }}
                    />
                    <span className={isHidden ? "line-through" : ""}>{key}</span>
                </button>
                )
            })}
            </div>
        )
        }
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={sortedRows} margin={{ top: 20, right: 20, left: 0, bottom: 80 }}>
          <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.1)" />
          <XAxis dataKey="channel" interval={0} angle={-45} textAnchor="end" height={90} tick={{ fill: "white", fontSize: 11 }} />
          <YAxis tick={{ fill: "white" }} />
          <Tooltip content={CustomTooltip as any} cursor={{ fill: "rgba(255,255,255,0.08)" }} />
          {seriesKeys.map((key) => (
            <Bar
                key={key}
                dataKey={key}
                stackId="membership"
                fill={colorFor(key, tierLabels)}
                hide={hidden.has(key)}
                name={key}
                animationDuration={250}
                animationEasing="ease-out"
                />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </ChartShell>
  )
}
