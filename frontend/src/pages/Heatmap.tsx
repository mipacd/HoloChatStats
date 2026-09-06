import { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import * as d3 from "d3"
import { Loader2 } from "lucide-react"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"
import { MonthPicker } from "@/components/ui/month-picker"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
const CHANNEL_GROUPS: Record<string, string> = {
  "Hololive EN":
    "Calli,Kiara,Ina,Gura,Amelia,Irys,Fauna,Mumei,Kronii,Baelz,Shiori,Bijou,Nerissa,FuwaMoco,Elizabeth,Gigi,Cecilia,Raora",
  "Hololive JP & ReGloss":
    "Sora,Roboco,Suisei,Miko,Azki,Haato,Fubuki,Aki,Matsuri,Choco,Ayame,Subaru,Shion,Korone,Mio,Okayu,Pekora,Marine,Noel,Flare,Towa,Watame,Kanata,Luna,Nene,Lamy,Polka,Botan,Koyori,Laplus,Chloe,Lui,Iroha,Hajime,Raden,Ao,Ririka,Kanade,Riona,Vivi,Su,Chihaya,Niko",
  "Hololive ID": "Iofi,Risu,Moona,Anya,Reine,Ollie,Kobo,Zeta,Kaela",
  "Indie (EN)": "Dooby,Nimi,Saba,Mint,Dokibird",
  "Indie (JP)": "Sakuna,Rica,Ruka,Rei,Roa",
  "Hololive EN + Select Indies":
    "Calli,Kiara,Ina,Gura,Amelia,Irys,Fauna,Mumei,Kronii,Baelz,Shiori,Bijou,Nerissa,FuwaMoco,Elizabeth,Gigi,Cecilia,Raora,Dooby,Nimi,Saba",
}
const COLOR_SCALE_ARRAY = [
  "#23171b","#271a28","#2b1c33","#2f1e3f","#32204a","#362354","#39255f","#3b2768","#3e2a72","#402c7b",
  "#422f83","#44318b","#453493","#46369b","#4839a2","#493ca8","#493eaf","#4a41b5","#4a44bb","#4b46c0",
  "#4b49c5","#4b4cca","#4b4ecf","#4b51d3","#4a54d7","#4a56db","#4959de","#495ce2","#485fe5","#4761e7",
  "#4664ea","#4567ec","#446aee","#446df0","#426ff2","#4172f3","#4075f5","#3f78f6","#3e7af7","#3d7df7",
  "#3c80f8","#3a83f9","#3985f9","#3888f9","#378bf9","#368df9","#3590f8","#3393f8","#3295f7","#3198f7",
  "#309bf6","#2f9df5","#2ea0f4","#2da2f3","#2ca5f1","#2ba7f0","#2aaaef","#2aaced","#29afec","#28b1ea",
  "#28b4e8","#27b6e6","#27b8e5","#26bbe3","#26bde1","#26bfdf","#25c1dc","#25c3da","#25c6d8","#25c8d6",
  "#25cad3","#25ccd1","#25cecf","#26d0cc","#26d2ca","#26d4c8","#27d6c5","#27d8c3","#28d9c0","#29dbbe",
  "#29ddbb","#2adfb8","#2be0b6","#2ce2b3","#2de3b1","#2ee5ae","#30e6ac","#31e8a9","#32e9a6","#34eba4",
  "#35eca1","#37ed9f","#39ef9c","#3af09a","#3cf197","#3ef295","#40f392","#42f490","#44f58d","#46f68b",
  "#48f788","#4af786","#4df884","#4ff981","#51fa7f","#54fa7d","#56fb7a","#59fb78","#5cfc76","#5efc74",
  "#61fd71","#64fd6f","#66fd6d","#69fd6b","#6cfd69","#6ffe67","#72fe65","#75fe63","#78fe61","#7bfe5f",
  "#7efd5d","#81fd5c","#84fd5a","#87fd58","#8afc56","#8dfc55","#90fb53","#93fb51","#96fa50","#99fa4e",
  "#9cf94d","#9ff84b","#a2f84a","#a6f748","#a9f647","#acf546","#aff444","#b2f343","#b5f242","#b8f141",
  "#bbf03f","#beef3e","#c1ed3d","#c3ec3c","#c6eb3b","#c9e93a","#cce839","#cfe738","#d1e537","#d4e336",
  "#d7e235","#d9e034","#dcdf33","#dedd32","#e0db32","#e3d931","#e5d730","#e7d52f","#e9d42f","#ecd22e",
  "#eed02d","#f0ce2c","#f1cb2c","#f3c92b","#f5c72b","#f7c52a","#f8c329","#fac029","#fbbe28","#fdbc28",
  "#feb927","#ffb727","#ffb526","#ffb226","#ffb025","#ffad25","#ffab24","#ffa824","#ffa623","#ffa323",
  "#ffa022","#ff9e22","#ff9b21","#ff9921","#ff9621","#ff9320","#ff9020","#ff8e1f","#ff8b1f","#ff881e",
  "#ff851e","#ff831d","#ff801d","#ff7d1d","#ff7a1c","#ff781c","#ff751b","#ff721b","#ff6f1a","#fd6c1a",
  "#fc6a19","#fa6719","#f96418","#f76118","#f65f18","#f45c17","#f25916","#f05716","#ee5415","#ec5115",
  "#ea4f14","#e84c14","#e64913","#e44713","#e24412","#df4212","#dd3f11","#da3d10","#d83a10","#d5380f",
  "#d3360f","#d0330e","#ce310d","#cb2f0d","#c92d0c","#c62a0b","#c3280b","#c1260a","#be2409","#bb2309",
  "#b92108","#b61f07","#b41d07","#b11b06","#af1a05","#ac1805","#aa1704","#a81604","#a51403","#a31302",
  "#a11202","#9f1101","#9d1000","#9b0f00","#980e00","#960d00","#950c00","#940c00","#930c00","#920c00",
  "#910b00","#910c00","#900c00","#900c00","#900c00",
]
interface MatrixData {
  labels: string[]
  matrix: number[][]
}
export default function Heatmap() {
  const { t, i18n } = useTranslation()
  const containerRef = useRef<HTMLDivElement>(null)
  const dataRef = useRef<MatrixData | null>(null)
  const membersOnlyRef = useRef(false)
  const [group, setGroup] = useState(Object.keys(CHANNEL_GROUPS)[0])
  const [month, setMonth] = useState(() => {
    const d = new Date()
    d.setMonth(d.getMonth() - 1)
    return d.toISOString().slice(0, 7)
  })
  const [membersOnly, setMembersOnly] = useState(false)
  const [loading, setLoading] = useState(false)
  const [tooltip, setTooltip] = useState({ x: 0, y: 0, text: "", visible: false })
  useEffect(() => {
    membersOnlyRef.current = membersOnly
  }, [membersOnly])
  const drawHeatmap = useCallback(
    (data: MatrixData | null) => {
      const container = containerRef.current
      if (!container) return
      d3.select(container).select("svg").remove()
      if (!data || data.labels.length === 0) return
      const { labels, matrix } = data
      const transposed = matrix[0].map((_, c) => matrix.map((row) => row[c]))
      const rect = container.getBoundingClientRect()
      const margin = { top: 20, right: 20, bottom: 100, left: 90 }
      const width = Math.max(rect.width - margin.left - margin.right, 50)
      const height = Math.max(rect.height - margin.top - margin.bottom, 50)
      const svg = d3
        .select(container)
        .append("svg")
        .attr("width", rect.width)
        .attr("height", rect.height)
        .attr("viewBox", `0 0 ${rect.width} ${rect.height}`)
        .attr("preserveAspectRatio", "xMidYMid meet")
      const g = svg.append("g").attr("transform", `translate(${margin.left},${margin.top})`)
      const xScale = d3.scaleBand().range([0, width]).domain(labels).padding(0.05)
      const yScale = d3.scaleBand().range([height, 0]).domain(labels).padding(0.05)
      const colors = d3.scaleQuantize<string>().domain([0, 60]).range(COLOR_SCALE_ARRAY)
      const flat: { row: number; col: number; value: number }[] = []
      transposed.forEach((row, i) => row.forEach((value, j) => flat.push({ row: i, col: j, value })))
      const audienceType = membersOnlyRef.current ? t("members") : t("chatters")
      const cellW = xScale.bandwidth()
      const cellH = yScale.bandwidth()
      const fontSize = Math.min(16, Math.max(8, Math.min(cellW, cellH) * 0.26))
      g.selectAll(".cell")
        .data(flat)
        .enter()
        .append("rect")
        .attr("class", "cell")
        .attr("x", (d) => xScale(labels[d.col])!)
        .attr("y", (d) => yScale(labels[d.row])!)
        .attr("width", cellW)
        .attr("height", cellH)
        .style("fill", (d) => (d.row === d.col ? "#1f1f1f" : colors(d.value)))
        .on("mousemove", (event: MouseEvent, d) => {
          if (d.row === d.col) {
            setTooltip((tp) => ({ ...tp, visible: false }))
            return
          }
          const rowLabel = labels[d.col]
          const colLabel = labels[d.row]
          const text = t("Percentage of {{row}}'s {{audience}} in {{col}}'s chat: {{value}}%", {
            defaultValue: "Percentage of {{row}}'s {{audience}} in {{col}}'s chat: {{value}}%",
            row: rowLabel,
            audience: audienceType,
            col: colLabel,
            value: d.value,
          })
          setTooltip({ visible: true, x: event.clientX, y: event.clientY, text })
        })
        .on("mouseout", () => setTooltip((tp) => ({ ...tp, visible: false })))
      g.selectAll(".cell-text")
        .data(flat)
        .enter()
        .append("text")
        .attr("x", (d) => xScale(labels[d.col])! + cellW / 2)
        .attr("y", (d) => yScale(labels[d.row])! + cellH / 2)
        .attr("text-anchor", "middle")
        .attr("dominant-baseline", "middle")
        .style("font-size", `${fontSize}px`)
        .style("fill", "black")
        .style("pointer-events", "none")
        .text((d) => (d.row !== d.col && d.value > 0 ? `${d.value.toFixed(1)}%` : ""))
      g.append("g")
        .attr("transform", `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .selectAll("text")
        .attr("transform", "translate(-10,0)rotate(-45)")
        .style("text-anchor", "end")
        .style("fill", "white")
      g.append("g")
        .call(d3.axisLeft(yScale))
        .selectAll("text")
        .style("fill", "white")
    },
    [t]
  )
  const fetchData = useCallback(async () => {
    const channels = CHANNEL_GROUPS[group]
    if (!group || !channels || !month) return
    setLoading(true)
    try {
      const res = await api.get("/get_common_users_matrix", {
        params: { month, channels, members_only: membersOnly },
      })
      const data = res.data
      if (!data || data.error) {
        dataRef.current = null
        drawHeatmap(null)
        return
      }
      const validIdx: number[] = []
      data.labels.forEach((_: string, i: number) => {
        if (data.matrix[i][i] > 0) validIdx.push(i)
      })
      const filteredLabels = data.labels.filter((_: string, i: number) => validIdx.includes(i))
      const filteredMatrix = data.matrix
        .filter((_: number[], i: number) => validIdx.includes(i))
        .map((row: number[]) => row.filter((_: number, j: number) => validIdx.includes(j)))
      const filtered = { labels: filteredLabels, matrix: filteredMatrix }
      dataRef.current = filtered.labels.length > 0 ? filtered : null
      drawHeatmap(dataRef.current)
    } catch (err) {
      console.error(err)
      dataRef.current = null
      drawHeatmap(null)
    } finally {
      setLoading(false)
    }
  }, [group, month, membersOnly, drawHeatmap])
  // Redraw on container resize (this is the responsiveness fix)
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    const ro = new ResizeObserver(() => {
      if (dataRef.current) drawHeatmap(dataRef.current)
    })
    ro.observe(container)
    return () => ro.disconnect()
  }, [drawHeatmap])
  useEffect(() => {
    fetchData()
  }, [fetchData])
  useEffect(() => {
    registerEriContext(() => ({
      page: "heatmap",
      endpoint: "/api/get_common_users_matrix",
      parameters: { group, month, members_only: membersOnly },
      description: `Viewing common user heatmap for ${group} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [group, month, membersOnly])
  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-2xl font-bold text-center">{t("Common User Heatmap")}</h2>
      <div className="flex flex-wrap items-end justify-center gap-4">
        <div className="space-y-1.5">
          <Label>{t("Channel Group:")}</Label>
          <Select value={group} onValueChange={setGroup}>
            <SelectTrigger className="w-[220px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {Object.keys(CHANNEL_GROUPS).map((g) => (
                <SelectItem key={g} value={g}>
                  {g}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="space-y-1.5">
          <Label>{t("Month:")}</Label>
          <MonthPicker value={month} onChange={setMonth} locale={i18n.language} />
        </div>
        <div className="flex items-center gap-2 pb-2">
          <Switch id="membersOnly" checked={membersOnly} onCheckedChange={setMembersOnly} />
          <Label htmlFor="membersOnly">{t("Members Only")}</Label>
        </div>
      </div>
      <div
        ref={containerRef}
        className="relative"
        style={{
            // full-bleed: break out of the max-w-7xl parent and span 95vw, centered
            width: "95vw",
            marginLeft: "calc(50% - 47.5vw)",
            // leave room for navbar + controls (top) and the Eri widget (bottom ~120px)
            height: "calc(100vh - 260px)",
            minHeight: 400,
        }}
        >
        {loading && (
            <div className="absolute inset-0 flex items-center justify-center bg-background/50 z-10">
            <Loader2 className="h-10 w-10 animate-spin text-primary" />
            </div>
        )}
        </div>
      {tooltip.visible && (
        <div
          className="fixed z-50 pointer-events-none rounded-md border border-border bg-popover px-3 py-2 text-sm text-popover-foreground shadow-lg max-w-xs"
          style={{ left: tooltip.x + 15, top: tooltip.y - 28 }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  )
}