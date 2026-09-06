import { useEffect, useMemo, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { io, type Socket } from "socket.io-client"
import { Bar, BarChart, CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig,
} from "@/components/ui/chart"
import { WorldMap } from "@/components/metrics/WorldMap"
import { alpha2ToName } from "@/lib/country-codes"
import { registerEriContext } from "@/components/eri/eri-context"
interface Metrics {
  unique_visitors: Record<string, number>
  page_views: Record<string, number>
  country_visits: Record<string, number>
  cache_data: Record<string, { cache_hits: number; cache_misses: number }>
}
function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="pt-6 text-center">
        <div className="text-xs uppercase tracking-wide text-muted-foreground">{label}</div>
        <div className="text-2xl font-bold text-[#4ecdc4] mt-1">{value}</div>
      </CardContent>
    </Card>
  )
}
export default function SiteMetrics() {
  const { t } = useTranslation()
  const [metrics, setMetrics] = useState<Metrics | null>(null)
  const pendingRef = useRef<Metrics | null>(null)
  const socketRef = useRef<Socket | null>(null)
  useEffect(() => {
    registerEriContext(() => ({ page: "site_metrics", description: "Viewing live site metrics" }))
    return () => registerEriContext(null)
  }, [])
  useEffect(() => {
    const socket = io({
      transports: ["websocket"],
      reconnection: true,
      reconnectionAttempts: 5,
      reconnectionDelay: 2000,
      timeout: 5000,
    })
    socketRef.current = socket
    socket.on("connect", () => socket.emit("request_update"))
    socket.on("metrics_update", (data: string) => {
      const parsed: Metrics = JSON.parse(data)
      if (document.hidden) {
        pendingRef.current = parsed
      } else {
        setMetrics(parsed)
      }
    })
    const onVisible = () => {
      if (!document.hidden && pendingRef.current) {
        setMetrics(pendingRef.current)
        pendingRef.current = null
      }
    }
    document.addEventListener("visibilitychange", onVisible)
    return () => {
      document.removeEventListener("visibilitychange", onVisible)
      socket.disconnect()
    }
  }, [])
  const derived = useMemo(() => {
    if (!metrics) return null
    const visitors = Object.entries(metrics.unique_visitors)
      .sort((a, b) => +new Date(a[0]) - +new Date(b[0]))
    const monthlyTotal = visitors.reduce((s, [, v]) => s + (Number(v) || 0), 0)
    const historical = visitors.slice(0, -1)
    const avgDaily = historical.length
      ? Math.round(historical.reduce((s, [, v]) => s + (Number(v) || 0), 0) / historical.length)
      : 0
    const totalPageViews = Object.values(metrics.page_views).reduce((s, v) => s + (Number(v) || 0), 0)
    const countriesReached = Object.keys(metrics.country_visits).length
    const visitorData = visitors.map(([date, count]) => ({ date, visitors: Number(count) }))
    const pageData = Object.entries(metrics.page_views)
      .sort((a, b) => b[1] - a[1])
      .map(([page, views]) => ({ page, views: Number(views) }))
    const cacheData = Object.entries(metrics.cache_data)
      .sort((a, b) => +new Date(a[0]) - +new Date(b[0]))
      .map(([date, v]) => ({ date, hits: Number(v.cache_hits || 0), misses: Number(v.cache_misses || 0) }))
    const topCountries = Object.entries(metrics.country_visits)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([code, count]) => ({ name: alpha2ToName[code] || code, count: Number(count) }))
    return { monthlyTotal, avgDaily, totalPageViews, countriesReached, visitorData, pageData, cacheData, topCountries }
  }, [metrics])
  const visitorsConfig = { visitors: { label: t("Unique Visitors"), color: "rgba(59,130,246,1)" } } satisfies ChartConfig
  const pagesConfig = { views: { label: t("Page Views"), color: "rgba(75,192,192,0.7)" } } satisfies ChartConfig
  const cacheConfig = {
    hits: { label: t("Cache Hits"), color: "rgba(75,192,192,0.7)" },
    misses: { label: t("Cache Misses"), color: "rgba(255,99,132,0.7)" },
  } satisfies ChartConfig
  return (
    <div className="flex flex-col gap-4 p-4">
      <h2 className="text-2xl font-bold text-center">{t("Live Site Metrics")}</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label={t("Unique Visitors (Last 30 Days)")} value={derived ? derived.monthlyTotal.toLocaleString() : "–"} />
        <StatCard label={t("Avg. Daily Visitors")} value={derived ? derived.avgDaily.toLocaleString() : "–"} />
        <StatCard label={t("Total Page Views")} value={derived ? derived.totalPageViews.toLocaleString() : "–"} />
        <StatCard label={t("Countries Reached")} value={derived ? derived.countriesReached.toLocaleString() : "–"} />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader><CardTitle className="text-base">{t("Total Page Views (Last 30 Days)")}</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px] overflow-x-auto">
              <div className="h-full" style={{ minWidth: derived ? Math.max(derived.pageData.length * 40, 100) : "100%" }}>
                <ChartContainer config={pagesConfig} className="h-full w-full !aspect-auto">
                  <BarChart data={derived?.pageData ?? []} margin={{ top: 10, right: 10, left: 0, bottom: 70 }}>
                    <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.1)" />
                    <XAxis dataKey="page" interval={0} angle={-45} textAnchor="end" height={80} tick={{ fill: "white", fontSize: 10 }} />
                    <YAxis tick={{ fill: "white" }} />
                    <ChartTooltip content={<ChartTooltipContent />} />
                    <Bar dataKey="views" fill="var(--color-views)" radius={[2, 2, 0, 0]} />
                  </BarChart>
                </ChartContainer>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">{t("Unique Visitors Over Time")}</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ChartContainer config={visitorsConfig} className="h-full w-full !aspect-auto">
                <LineChart data={derived?.visitorData ?? []} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="date" tick={{ fill: "white", fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis tick={{ fill: "white" }} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Line type="linear" dataKey="visitors" stroke="var(--color-visitors)" strokeWidth={2} dot={false} />
                </LineChart>
              </ChartContainer>
            </div>
          </CardContent>
        </Card>
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle className="text-base">{t("Unique Visitors by Country (Last 30 Days)")}</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[460px]">
              <WorldMap data={metrics?.country_visits ?? {}} />
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">{t("Cache Data (Hits vs Misses)")}</CardTitle></CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <ChartContainer config={cacheConfig} className="h-full w-full !aspect-auto">
                <BarChart data={derived?.cacheData ?? []} margin={{ top: 10, right: 20, left: 0, bottom: 40 }}>
                  <CartesianGrid vertical={false} stroke="rgba(255,255,255,0.1)" />
                  <XAxis dataKey="date" tick={{ fill: "white", fontSize: 10 }} angle={-45} textAnchor="end" height={60} />
                  <YAxis tick={{ fill: "white" }} />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <ChartLegend content={<ChartLegendContent />} />
                  <Bar dataKey="hits" stackId="c" fill="var(--color-hits)" />
                  <Bar dataKey="misses" stackId="c" fill="var(--color-misses)" />
                </BarChart>
              </ChartContainer>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="text-base">{t("Top countries:")}</CardTitle></CardHeader>
          <CardContent>
            {derived && derived.topCountries.length ? (
              <ol className="space-y-2">
                {derived.topCountries.map((c, i) => (
                  <li key={c.name} className="flex justify-between">
                    <span><span className="text-muted-foreground mr-2">{i + 1}.</span>{c.name}</span>
                    <span className="font-mono text-[#4ecdc4]">{c.count.toLocaleString()}</span>
                  </li>
                ))}
              </ol>
            ) : (
              <p className="text-muted-foreground">–</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}