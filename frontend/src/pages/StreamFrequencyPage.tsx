import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { X, Info, Settings2, ChevronLeft, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { Button } from "@/components/ui/button"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import {
  Tooltip, TooltipContent, TooltipProvider, TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Popover, PopoverContent, PopoverTrigger,
} from "@/components/ui/popover"
import {
  Tabs, TabsContent, TabsList, TabsTrigger,
} from "@/components/ui/tabs"
import { ChartShell } from "@/components/charts/ChartShell"
import { TimezoneCombobox } from "@/components/TimezoneCombobox"
import { api } from "@/lib/api"
import { downloadCSV } from "@/lib/chart-export"
import { registerEriContext } from "@/components/eri/eri-context"
/* ══════════════════════════════════════════════════════════════
   Shared types & helpers
   ══════════════════════════════════════════════════════════════ */
type Mode = "span" | "start"
type Resolution = 60 | 30
type WeekStart = "sunday" | "monday"
interface ChannelFrequency {
  channel: string
  frequency: Record<string, number>
  dayFrequency: Record<string, number>
}
/* ── colour ────────────────────────────────────────────────── */
interface ColorStop {
  t: number; r: number; g: number; b: number
}
const STOPS: ColorStop[] = [
  { t: 0, r: 59, g: 130, b: 246 },
  { t: 0.33, r: 34, g: 197, b: 94 },
  { t: 0.66, r: 234, g: 179, b: 8 },
  { t: 1, r: 239, g: 68, b: 68 },
]
const ZERO_COLOR = "#374151"
/**
 * Absolute reference for calendar circle colour + size.
 * A day with >= this many streams shows max colour/size.
 */
const CAL_ABS_MAX = 4
function heatColor(value: number, max: number): string {
  if (value === 0 || max === 0) return ZERO_COLOR
  const t = Math.min(value / max, 1)
  let lo = STOPS[0]
  let hi = STOPS[STOPS.length - 1]
  for (let i = 0; i < STOPS.length - 1; i++) {
    if (t >= STOPS[i].t && t <= STOPS[i + 1].t) {
      lo = STOPS[i]
      hi = STOPS[i + 1]
      break
    }
  }
  const p = (t - lo.t) / (hi.t - lo.t || 1)
  return `rgb(${Math.round(lo.r + (hi.r - lo.r) * p)},${Math.round(
    lo.g + (hi.g - lo.g) * p
  )},${Math.round(lo.b + (hi.b - lo.b) * p)})`
}
/* ── labels ────────────────────────────────────────────────── */
const ALL_DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
function getDayOrder(ws: WeekStart): number[] {
  return ws === "monday" ? [1, 2, 3, 4, 5, 6, 0] : [0, 1, 2, 3, 4, 5, 6]
}
function slotLabel(index: number, res: Resolution, use12h: boolean): string {
  const mins = index * res
  const h = Math.floor(mins / 60)
  const m = mins % 60
  if (use12h) {
    const sfx = h < 12 ? "AM" : "PM"
    const h12 = h === 0 ? 12 : h > 12 ? h - 12 : h
    return m === 0
      ? `${h12}${sfx}`
      : `${h12}:${String(m).padStart(2, "0")}${sfx}`
  }
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`
}
function getUserTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone
  } catch {
    return "UTC"
  }
}
/* ══════════════════════════════════════════════════════════════
   Shared sub-components
   ══════════════════════════════════════════════════════════════ */
/* ── HeatmapRow ────────────────────────────────────────────── */
interface HeatmapRowProps {
  items: { key: string; label: string; count: number }[]
  max: number
  blockClass?: string
}
function HeatmapRow({ items, max, blockClass = "" }: HeatmapRowProps) {
  const { t } = useTranslation()
  return (
    <div className="flex flex-col gap-1">
      <div className="flex w-full gap-[2px]">
        {items.map((d) => (
          <Tooltip key={d.key}>
            <TooltipTrigger asChild>
              <div
                className={cn(
                  "flex-1 rounded-sm cursor-default transition-colors",
                  "flex items-center justify-center text-[10px] font-medium text-black select-none",
                  blockClass
                )}
                style={{ backgroundColor: heatColor(d.count, max) }}
              >
                {d.count > 0 ? d.count : ""}
              </div>
            </TooltipTrigger>
            <TooltipContent side="top">
              <p className="font-semibold">{d.label}</p>
              <p>
                {d.count} {d.count === 1 ? t("stream") : t("streams")}
              </p>
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
      <div className="flex w-full gap-[2px]">
        {items.map((d) => (
          <span
            key={d.key}
            className="flex-1 text-center text-[10px] leading-tight text-muted-foreground select-none"
          >
            {d.label}
          </span>
        ))}
      </div>
    </div>
  )
}
/* ── Heatmap colour legend ─────────────────────────────────── */
function ColorLegend() {
  const { t } = useTranslation()
  return (
    <div className="flex items-center justify-center gap-2 text-xs text-muted-foreground">
      <span>0</span>
      <div className="flex h-3 rounded overflow-hidden" style={{ width: 200 }}>
        <div
          className="h-full"
          style={{ width: 14, backgroundColor: ZERO_COLOR }}
        />
        {Array.from({ length: 20 }, (_, i) => (
          <div
            key={i}
            className="flex-1 h-full"
            style={{ backgroundColor: heatColor(i + 1, 20) }}
          />
        ))}
      </div>
      <span>{t("Max")}</span>
    </div>
  )
}
/* ── Calendar colour legend (absolute / discrete) ──────────── */
function CalendarLegend() {
  const counts = [1, 2, 3, 4]
  return (
    <div className="flex items-center gap-4 text-xs text-muted-foreground">
      {counts.map((n) => (
        <div key={n} className="flex items-center gap-1.5">
          <div
            className="rounded-full"
            style={{
              width: 12,
              height: 12,
              backgroundColor: heatColor(n, CAL_ABS_MAX),
            }}
          />
          <span>{n === CAL_ABS_MAX ? `${n}+` : n}</span>
        </div>
      ))}
    </div>
  )
}
/* ── WeekStart toggle (shared between both Advanced menus) ─── */
function WeekStartToggle({
  weekStart,
  setWeekStart,
}: {
  weekStart: WeekStart
  setWeekStart: (w: WeekStart) => void
}) {
  const { t } = useTranslation()
  return (
    <div className="flex items-center justify-between">
      <Label htmlFor="week-start">{t("Start week on Monday")}</Label>
      <Switch
        id="week-start"
        checked={weekStart === "monday"}
        onCheckedChange={(v) => setWeekStart(v ? "monday" : "sunday")}
      />
    </div>
  )
}
/* ══════════════════════════════════════════════════════════════
   ChannelCard (heatmap tab)
   ══════════════════════════════════════════════════════════════ */
interface ChannelCardProps {
  data: ChannelFrequency
  use12h: boolean
  resolution: Resolution
  weekStart: WeekStart
  onRemove: () => void
}
function ChannelCard({
  data,
  use12h,
  resolution,
  weekStart,
  onRemove,
}: ChannelCardProps) {
  const { t } = useTranslation()
  const numSlots = (24 * 60) / resolution
  const dayOrder = useMemo(() => getDayOrder(weekStart), [weekStart])
  const hourData = useMemo(
    () =>
      Array.from({ length: numSlots }, (_, i) => ({
        key: `s${i}`,
        label: slotLabel(i, resolution, use12h),
        count: data.frequency[String(i)] ?? 0,
      })),
    [data.frequency, use12h, resolution, numSlots]
  )
  const dayData = useMemo(
    () =>
      dayOrder.map((idx) => ({
        key: `d${idx}`,
        label: t(ALL_DAY_LABELS[idx]),
        count: data.dayFrequency[String(idx)] ?? 0,
      })),
    [data.dayFrequency, dayOrder, t]
  )
  const maxHour = useMemo(
    () => Math.max(0, ...hourData.map((d) => d.count)),
    [hourData]
  )
  const maxDay = useMemo(
    () => Math.max(0, ...dayData.map((d) => d.count)),
    [dayData]
  )
  return (
    <div className="relative rounded-lg border border-border bg-card p-4">
      <Button
        variant="ghost"
        size="icon"
        className="absolute top-2 right-2 h-6 w-6 text-muted-foreground hover:text-destructive"
        onClick={onRemove}
        aria-label={t("Remove {{channel}}", { channel: data.channel })}
      >
        <X className="h-4 w-4" />
      </Button>
      <h3 className="text-sm font-semibold mb-4 pr-8">{data.channel}</h3>
      <TooltipProvider delayDuration={0}>
        <div className="flex flex-col gap-5">
          <div className="mx-auto w-full max-w-md">
            <p className="text-xs text-muted-foreground text-center mb-2">
              {t("Day of Week")}
            </p>
            <HeatmapRow items={dayData} max={maxDay} blockClass="h-10" />
          </div>
          <div>
            <p className="text-xs text-muted-foreground text-center mb-2">
              {t("Hour of Day")}
            </p>
            <HeatmapRow items={hourData} max={maxHour} blockClass="h-10" />
          </div>
        </div>
      </TooltipProvider>
    </div>
  )
}
/* ══════════════════════════════════════════════════════════════
   MonthGrid (calendar tab)
   ══════════════════════════════════════════════════════════════ */
interface MonthGridProps {
  year: number
  month: number // 0-indexed
  calendar: Record<string, number>
  weekStart: WeekStart
  cell: number // px size of one day cell — drives all sizing
}
/**
 * Circle diameter as a % of the cell, by stream count:
 *
 *  1 stream  →  55 %   snugly encircles the date number
 *  2 streams →  83 %   comfortably backgrounds a two-digit number
 *  3 streams → 112 %   extends slightly beyond the cell edge
 *  4+streams → 140 %   reaches the edge of the neighbouring number
 *
 * Counts above CAL_ABS_MAX are clamped to max.
 */
function calDotPct(count: number): number {
  if (count <= 0) return 0
  const c = Math.min(count, CAL_ABS_MAX)
  return 55 + 85 * ((c - 1) / (CAL_ABS_MAX - 1))
}
function MonthGrid({ year, month, calendar, weekStart, cell }: MonthGridProps) {
  const { t } = useTranslation()
  const monthName = new Date(year, month, 1).toLocaleString("default", {
    month: "short",
  })
  const dayHeaders =
    weekStart === "monday"
      ? ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
      : ["Su", "Mo", "Tu", "We", "Th", "Fr", "Sa"]
  const daysInMonth = new Date(year, month + 1, 0).getDate()
  let startDow = new Date(year, month, 1).getDay()
  if (weekStart === "monday") startDow = (startDow + 6) % 7
  const cells: (number | null)[] = new Array(startDow).fill(null)
  for (let d = 1; d <= daysInMonth; d++) cells.push(d)
  while (cells.length % 7 !== 0) cells.push(null)
  return (
    <div className="flex flex-col items-center">
      {/* month name */}
      <div
        className="font-semibold flex items-center justify-center"
        style={{ height: cell * 1.3, fontSize: cell * 0.8 }}
      >
        {monthName}
      </div>
      <div className="grid grid-cols-7">
        {/* weekday headers */}
        {dayHeaders.map((h, i) => (
          <div
            key={`h${i}`}
            className="flex items-center justify-center text-muted-foreground font-medium"
            style={{ width: cell, height: cell * 0.9, fontSize: cell * 0.48 }}
          >
            {h}
          </div>
        ))}
        {/* day cells */}
        {cells.map((day, i) => {
          if (day === null)
            return <div key={`e${i}`} style={{ width: cell, height: cell }} />
          const dateStr = `${year}-${String(month + 1).padStart(2, "0")}-${String(day).padStart(2, "0")}`
          const count = calendar[dateStr] ?? 0
          const diaPct = calDotPct(count)
          const cellEl = (
            <div
              className="relative flex items-center justify-center"
              style={{ width: cell, height: cell }}
            >
              {count > 0 && (
                <div
                  className="absolute rounded-full"
                  style={{
                    width: `${diaPct}%`,
                    height: `${diaPct}%`,
                    backgroundColor: heatColor(count, CAL_ABS_MAX),
                  }}
                />
              )}
              <span
                className={cn(
                  "relative z-10",
                  count > 0
                    ? "text-black font-semibold"
                    : "text-muted-foreground"
                )}
                style={{ fontSize: cell * 0.46 }}
              >
                {day}
              </span>
            </div>
          )
          if (count === 0) return <div key={dateStr}>{cellEl}</div>
          return (
            <Tooltip key={dateStr}>
              <TooltipTrigger asChild>{cellEl}</TooltipTrigger>
              <TooltipContent side="top">
                <p className="font-semibold">
                  {new Date(year, month, day).toLocaleDateString(undefined, {
                    month: "long",
                    day: "numeric",
                    year: "numeric",
                  })}
                </p>
                <p>
                  {count} {count === 1 ? t("stream") : t("streams")}
                </p>
              </TooltipContent>
            </Tooltip>
          )
        })}
      </div>
    </div>
  )
}
/* ══════════════════════════════════════════════════════════════
   View props shared by both tabs
   ══════════════════════════════════════════════════════════════ */
interface ViewProps {
  channelNames: string[]
  timezone: string
  setTimezone: (tz: string) => void
  weekStart: WeekStart
  setWeekStart: (ws: WeekStart) => void
  isActive: boolean
}
/* ══════════════════════════════════════════════════════════════
   HeatmapView
   ══════════════════════════════════════════════════════════════ */
function HeatmapView({
  channelNames,
  timezone,
  setTimezone,
  weekStart,
  setWeekStart,
  isActive,
}: ViewProps) {
  const { t } = useTranslation()
  const [pendingChannel, setPendingChannel] = useState("")
  const [mode, setMode] = useState<Mode>("span")
  const [resolution, setResolution] = useState<Resolution>(60)
  const [excludeShorts, setExcludeShorts] = useState(true)
  const [use12h, setUse12h] = useState(true)
  const [entries, setEntries] = useState<ChannelFrequency[]>([])
  const [loading, setLoading] = useState(false)
  const [capturing, setCapturing] = useState(false)
  const addedSet = useMemo(
    () => new Set(entries.map((e) => e.channel)),
    [entries]
  )
  const available = useMemo(
    () => channelNames.filter((n) => !addedSet.has(n)),
    [channelNames, addedSet]
  )
  const fetchChannel = useCallback(
    async (name: string): Promise<ChannelFrequency | null> => {
      try {
        const res = await api.get("/get_stream_frequency", {
          params: {
            channel_name: name,
            timezone,
            mode,
            resolution: String(resolution),
            exclude_shorts: excludeShorts ? "1" : "0",
          },
        })
        return {
          channel: name,
          frequency: res.data?.frequency ?? {},
          dayFrequency: res.data?.day_frequency ?? {},
        }
      } catch {
        return null
      }
    },
    [timezone, mode, resolution, excludeShorts]
  )
  const handleAdd = useCallback(async () => {
    if (!pendingChannel || addedSet.has(pendingChannel)) return
    setLoading(true)
    const r = await fetchChannel(pendingChannel)
    if (r) setEntries((p) => [...p, r])
    setPendingChannel("")
    setLoading(false)
  }, [pendingChannel, addedSet, fetchChannel])
  const handleRemove = useCallback(
    (n: string) => setEntries((p) => p.filter((e) => e.channel !== n)),
    []
  )
  const currentChannels = useMemo(
    () => entries.map((e) => e.channel),
    [entries]
  )
  /* re-fetch all when shared controls change */
  useEffect(() => {
    if (currentChannels.length === 0) return
    let cancelled = false
    setLoading(true)
    Promise.all(currentChannels.map(fetchChannel)).then((r) => {
      if (cancelled) return
      setEntries(r.filter(Boolean) as ChannelFrequency[])
      setLoading(false)
    })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timezone, mode, resolution, excludeShorts])
  /* eri context */
  useEffect(() => {
    if (!isActive) return
    registerEriContext(() => ({
      page: "stream_frequency_heatmap",
      endpoint: "/api/get_stream_frequency",
      parameters: {
        channels: currentChannels.length ? currentChannels : "(none)",
        timezone,
        mode,
        resolution,
        exclude_shorts: excludeShorts,
      },
      description: `Heatmap: ${currentChannels.join(", ") || "(none)"} in ${timezone}`,
    }))
    return () => registerEriContext(null)
  }, [isActive, currentChannels, timezone, mode, resolution, excludeShorts])
  const hasData = entries.length > 0
  const numSlots = (24 * 60) / resolution
  const handleCSV = () => {
    const rows: (string | number)[][] = []
    for (const e of entries) {
      getDayOrder(weekStart).forEach((idx) =>
        rows.push([
          e.channel,
          "Day",
          ALL_DAY_LABELS[idx],
          e.dayFrequency[String(idx)] ?? 0,
        ])
      )
      Array.from({ length: numSlots }, (_, i) =>
        rows.push([
          e.channel,
          "Hour",
          slotLabel(i, resolution, use12h),
          e.frequency[String(i)] ?? 0,
        ])
      )
    }
    downloadCSV(
      "stream_frequency",
      ["Channel", "Type", "Slot", "Streams"],
      rows
    )
  }
  return (
    <ChartShell
      title={t("Stream Frequency — Heatmap")}
      infoText={t(
        "Compare how frequently channels stream during each day and hour over the past year (DST-aware)."
      )}
      loading={loading}
      hasData={hasData}
      chartMinWidth={resolution === 30 ? 960 : 640}
      onDownloadCSV={handleCSV}
      png={{
        title: `Stream Frequency Heatmap - ${
          currentChannels.join(", ") || "N/A"
        } - ${timezone} - holochatstats.info`,
        filename: "stream_frequency_heatmap",
        onBeforeCapture: () => setCapturing(true),
        onAfterCapture: () => setCapturing(false),
      }}
      controls={
        <div className="flex flex-wrap items-end gap-4">
          {/* Add channel */}
          <div className="space-y-1.5">
            <Label>{t("Add Channel:")}</Label>
            <div className="flex gap-2">
              <Select value={pendingChannel} onValueChange={setPendingChannel}>
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder={t("Select channel")} />
                </SelectTrigger>
                <SelectContent>
                  {available.map((n) => (
                    <SelectItem key={n} value={n}>
                      {n}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <Button
                onClick={handleAdd}
                disabled={!pendingChannel || loading}
              >
                {t("Add")}
              </Button>
            </div>
          </div>
          {/* Timezone */}
          <div className="space-y-1.5">
            <Label>{t("Timezone:")}</Label>
            <TimezoneCombobox
              value={timezone}
              onChange={setTimezone}
              className="w-[320px]"
            />
          </div>
          {/* Advanced */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" className="gap-2 mb-px">
                <Settings2 className="h-4 w-4" />
                {t("Advanced")}
              </Button>
            </PopoverTrigger>
            <PopoverContent
              align="start"
              side="bottom"
              className="w-72 flex flex-col gap-4"
              onOpenAutoFocus={(e) => e.preventDefault()}
            >
              {/* Span / Start */}
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-1.5">
                  <Label htmlFor="hm-mode">
                    {mode === "start" ? t("Start mode") : t("Span mode")}
                  </Label>
                  <TooltipProvider delayDuration={0}>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="text-muted-foreground cursor-help">
                          <Info className="h-3.5 w-3.5" />
                        </span>
                      </TooltipTrigger>
                      <TooltipContent side="right" className="max-w-[240px]">
                        <p className="mb-1">
                          <span className="font-semibold">{t("Span")}:</span>{" "}
                          {t("counts every slot a stream was live.")}
                        </p>
                        <p>
                          <span className="font-semibold">{t("Start")}:</span>{" "}
                          {t(
                            "counts only the slot the stream started in."
                          )}
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Switch
                  id="hm-mode"
                  checked={mode === "start"}
                  onCheckedChange={(v) => setMode(v ? "start" : "span")}
                />
              </div>
              {/* Resolution */}
              <div className="flex items-center justify-between">
                <Label htmlFor="hm-res">{t("30-min resolution")}</Label>
                <Switch
                  id="hm-res"
                  checked={resolution === 30}
                  onCheckedChange={(v) => setResolution(v ? 30 : 60)}
                />
              </div>
              {/* Exclude shorts */}
              <div className="flex items-center justify-between">
                <Label htmlFor="hm-shorts">
                  {t("Exclude short videos (<10 min)")}
                </Label>
                <Switch
                  id="hm-shorts"
                  checked={excludeShorts}
                  onCheckedChange={setExcludeShorts}
                />
              </div>
              {/* 24h clock */}
              <div className="flex items-center justify-between">
                <Label htmlFor="hm-clock">{t("24-hour clock")}</Label>
                <Switch
                  id="hm-clock"
                  checked={!use12h}
                  onCheckedChange={(v) => setUse12h(!v)}
                />
              </div>
              {/* Week start */}
              <WeekStartToggle
                weekStart={weekStart}
                setWeekStart={setWeekStart}
              />
            </PopoverContent>
          </Popover>
        </div>
      }
    >
      <div
        className={cn(
          "px-2 py-4 flex flex-col gap-6",
          !capturing && "overflow-y-auto max-h-[calc(100vh-280px)]"
        )}
      >
        {entries.map((e) => (
          <ChannelCard
            key={e.channel}
            data={e}
            use12h={use12h}
            resolution={resolution}
            weekStart={weekStart}
            onRemove={() => handleRemove(e.channel)}
          />
        ))}
        {entries.length === 0 && (
          <p className="text-center text-sm text-muted-foreground py-12">
            {t("Add a channel above to view its stream frequency.")}
          </p>
        )}
        {entries.length > 0 && <ColorLegend />}
      </div>
    </ChartShell>
  )
}
/* ══════════════════════════════════════════════════════════════
   CalendarView
   ══════════════════════════════════════════════════════════════ */
function CalendarView({
  channelNames,
  timezone,
  setTimezone,
  weekStart,
  setWeekStart,
  isActive,
}: ViewProps) {
  const { t } = useTranslation()
  const [channel, setChannel] = useState("")
  const [excludeShorts, setExcludeShorts] = useState(true)
  const [calendar, setCalendar] = useState<Record<string, number>>({})
  const [availableYears, setAvailableYears] = useState<number[]>([])
  const [year, setYear] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [, setCapturing] = useState(false)
  /* ── measure container → derive cell size ────────────────── */
  const measureRef = useRef<HTMLDivElement>(null)
  const [cell, setCell] = useState(18)
  useEffect(() => {
    const el = measureRef.current
    if (!el) return
    const GAP = 16
    const COLS = 4
    const ROWS = 3
    // Vertical "units" per month: name(1.3) + header(0.9) + 6 weeks = 8.2
    const MONTH_H_UNITS = 8.2
    const MONTH_W_UNITS = 7
    const compute = () => {
      const w = el.clientWidth
      const h = el.clientHeight
      if (w <= 0 || h <= 0) return
      const byW = (w - (COLS - 1) * GAP) / (COLS * MONTH_W_UNITS)
      const byH = (h - (ROWS - 1) * GAP) / (ROWS * MONTH_H_UNITS)
      setCell(Math.max(10, Math.floor(Math.min(byW, byH))))
    }
    compute()
    const ro = new ResizeObserver(compute)
    ro.observe(el)
    return () => ro.disconnect()
  }, [year])
  /* ── reset year when channel changes ─────────────────────── */
  const prevChannel = useRef(channel)
  useEffect(() => {
    if (channel !== prevChannel.current) {
      setYear(null)
      prevChannel.current = channel
    }
  }, [channel])
  /* ── fetch calendar data ─────────────────────────────────── */
  useEffect(() => {
    if (!channel) return
    let cancelled = false
    setLoading(true)
    api
      .get("/get_stream_calendar", {
        params: {
          channel_name: channel,
          timezone,
          year: year ?? "latest",
          exclude_shorts: excludeShorts ? "1" : "0",
        },
      })
      .then((res) => {
        if (cancelled) return
        setCalendar(res.data?.calendar ?? {})
        setAvailableYears(res.data?.available_years ?? [])
        const returnedYear = res.data?.year
        if (returnedYear != null) setYear(returnedYear)
      })
      .catch(() => {
        if (!cancelled) {
          setCalendar({})
          setAvailableYears([])
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [channel, timezone, year, excludeShorts])
  /* ── year navigation ─────────────────────────────────────── */
  const yearIdx =
    year !== null ? availableYears.indexOf(year) : -1
  const canPrev = yearIdx > 0
  const canNext =
    yearIdx >= 0 && yearIdx < availableYears.length - 1
  /* ── eri context ─────────────────────────────────────────── */
  useEffect(() => {
    if (!isActive) return
    registerEriContext(() => ({
      page: "stream_frequency_calendar",
      endpoint: "/api/get_stream_calendar",
      parameters: {
        channel: channel || "(none)",
        timezone,
        year,
        exclude_shorts: excludeShorts,
      },
      description: `Calendar: ${channel || "(none)"} ${year ?? ""} in ${timezone}`,
    }))
    return () => registerEriContext(null)
  }, [isActive, channel, timezone, year, excludeShorts])
  const hasData = Object.keys(calendar).length > 0
  const handleCSV = () => {
    const rows = Object.entries(calendar)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([d, c]) => [d, c])
    downloadCSV("stream_calendar", ["Date", "Streams"], rows)
  }
  return (
    <ChartShell
      title={t("Stream Frequency — Calendar")}
      infoText={t(
        "Displays the days a channel streamed across the calendar year (DST-aware)."
      )}
      loading={loading}
      hasData={hasData}
      onDownloadCSV={handleCSV}
      png={{
        title: `Stream Calendar - ${channel || "N/A"} - ${year ?? ""} - ${timezone} - holochatstats.info`,
        filename: "stream_calendar",
        onBeforeCapture: () => setCapturing(true),
        onAfterCapture: () => setCapturing(false),
      }}
      controls={
        <div className="flex flex-wrap items-end gap-4">
          {/* Channel (single select) */}
          <div className="space-y-1.5">
            <Label>{t("Channel:")}</Label>
            <Select
              value={channel}
              onValueChange={(v) => {
                setChannel(v)
                setYear(null)
              }}
            >
              <SelectTrigger className="w-[200px]">
                <SelectValue placeholder={t("Select channel")} />
              </SelectTrigger>
              <SelectContent>
                {channelNames.map((n) => (
                  <SelectItem key={n} value={n}>
                    {n}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          {/* Timezone */}
          <div className="space-y-1.5">
            <Label>{t("Timezone:")}</Label>
            <TimezoneCombobox
              value={timezone}
              onChange={setTimezone}
              className="w-[320px]"
            />
          </div>
          {/* Advanced */}
          <Popover>
            <PopoverTrigger asChild>
              <Button variant="outline" className="gap-2 mb-px">
                <Settings2 className="h-4 w-4" />
                {t("Advanced")}
              </Button>
            </PopoverTrigger>
            <PopoverContent
              align="start"
              side="bottom"
              className="w-72 flex flex-col gap-4"
              onOpenAutoFocus={(e) => e.preventDefault()}
            >
              <div className="flex items-center justify-between">
                <Label htmlFor="cal-shorts">
                  {t("Exclude short videos (<10 min)")}
                </Label>
                <Switch
                  id="cal-shorts"
                  checked={excludeShorts}
                  onCheckedChange={setExcludeShorts}
                />
              </div>
              <WeekStartToggle
                weekStart={weekStart}
                setWeekStart={setWeekStart}
              />
            </PopoverContent>
          </Popover>
        </div>
      }
    >
      <div className="h-full flex flex-col items-center gap-3 px-2 py-2">
        {/* year navigation */}
        {year !== null && (
          <div className="flex items-center gap-4 shrink-0">
            <Button
              variant="ghost"
              size="icon"
              disabled={!canPrev}
              onClick={() =>
                canPrev && setYear(availableYears[yearIdx - 1])
              }
            >
              <ChevronLeft className="h-5 w-5" />
            </Button>
            <span className="text-xl font-bold w-16 text-center">
              {year}
            </span>
            <Button
              variant="ghost"
              size="icon"
              disabled={!canNext}
              onClick={() =>
                canNext && setYear(availableYears[yearIdx + 1])
              }
            >
              <ChevronRight className="h-5 w-5" />
            </Button>
          </div>
        )}
        {/* months grid — sized to fill remaining height, centred */}
        {year !== null && (
          <div
            ref={measureRef}
            className="flex-1 w-full min-h-0 flex items-center justify-center overflow-hidden"
          >
            <TooltipProvider delayDuration={0} disableHoverableContent>
              <div className="grid grid-cols-4" style={{ gap: 16 }}>
                {Array.from({ length: 12 }, (_, m) => (
                  <MonthGrid
                    key={m}
                    year={year}
                    month={m}
                    calendar={calendar}
                    weekStart={weekStart}
                    cell={cell}
                  />
                ))}
              </div>
            </TooltipProvider>
          </div>
        )}
        {year === null && !loading && channel && (
          <p className="text-sm text-muted-foreground py-12">
            {t("No stream data found for this channel.")}
          </p>
        )}
        {!channel && (
          <p className="text-sm text-muted-foreground py-12">
            {t("Select a channel above to view its stream calendar.")}
          </p>
        )}
        {hasData && (
          <div className="shrink-0">
            <CalendarLegend />
          </div>
        )}
      </div>
    </ChartShell>
  )
}
/* ══════════════════════════════════════════════════════════════
   Page root — tabs
   ══════════════════════════════════════════════════════════════ */
export default function StreamFrequencyPage() {
  const { t } = useTranslation()
  /* shared across both tabs */
  const [channelNames, setChannelNames] = useState<string[]>([])
  const [timezone, setTimezone] = useState(getUserTimezone)
  const [weekStart, setWeekStart] = useState<WeekStart>("sunday")
  const [activeTab, setActiveTab] = useState("heatmap")
  useEffect(() => {
    api
      .get("/get_channel_names")
      .then((res) => {
        const raw = res.data
        const names: string[] = Array.isArray(raw)
          ? raw
          : raw?.data ?? raw?.channel_names ?? []
        setChannelNames(names.sort((a, b) => a.localeCompare(b)))
      })
      .catch(() => setChannelNames([]))
  }, [])
  return (
    <Tabs
      value={activeTab}
      onValueChange={setActiveTab}
      className="w-full"
    >
      <TabsList className="mx-auto flex w-fit mb-4">
        <TabsTrigger value="heatmap">{t("Heatmap")}</TabsTrigger>
        <TabsTrigger value="calendar">{t("Calendar")}</TabsTrigger>
      </TabsList>
      <TabsContent
        value="heatmap"
        forceMount
        className="data-[state=inactive]:hidden"
      >
        <HeatmapView
          channelNames={channelNames}
          timezone={timezone}
          setTimezone={setTimezone}
          weekStart={weekStart}
          setWeekStart={setWeekStart}
          isActive={activeTab === "heatmap"}
        />
      </TabsContent>
      <TabsContent
        value="calendar"
        forceMount
        className="data-[state=inactive]:hidden"
      >
        <CalendarView
          channelNames={channelNames}
          timezone={timezone}
          setTimezone={setTimezone}
          weekStart={weekStart}
          setWeekStart={setWeekStart}
          isActive={activeTab === "calendar"}
        />
      </TabsContent>
    </Tabs>
  )
}
