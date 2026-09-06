import { useEffect, useRef, useState } from "react"
import * as d3 from "d3"
import * as topojson from "topojson-client"
import { Plus, Minus, Home } from "lucide-react"
import { numToA2, numericToName } from "@/lib/country-codes"
const NO_VISITORS = "#2d3436"
const MIN_COLOR = "#4a6fa5"
const MID_COLOR = "#e17055"
const MAX_COLOR = "#d63031"
const FLASH_COLOR = "#ffeb3b"
function getCountryColor(visitors: number, maxVisitors: number) {
  if (!visitors) return NO_VISITORS
  if (maxVisitors <= 1) return MIN_COLOR
  const t = Math.log(visitors) / Math.log(maxVisitors)
  return t < 0.5
    ? d3.interpolateRgb(MIN_COLOR, MID_COLOR)(t * 2)
    : d3.interpolateRgb(MID_COLOR, MAX_COLOR)((t - 0.5) * 2)
}
const codeOf = (f: any) => numToA2[String(f.id).padStart(3, "0")] || null
const nameOf = (f: any) => numericToName[String(f.id).padStart(3, "0")] || "Unknown"
export function WorldMap({ data }: { data: Record<string, number> }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const dataRef = useRef(data)
  dataRef.current = data
  const firstApplied = useRef(false)
  const updateRef = useRef<((d: Record<string, number>, animate: boolean) => void) | null>(null)
  const zoomApi = useRef<{ in: () => void; out: () => void; reset: () => void } | null>(null)
  const [maxVisitors, setMaxVisitors] = useState(1)
  const [zoomPct, setZoomPct] = useState(100)
  useEffect(() => {
    const container = containerRef.current
    if (!container) return
    let mounted = true
    let prevData: Record<string, number> | null = null
    const svg = d3.select(container).append("svg")
      .attr("width", "100%").attr("height", "100%")
    const g = svg.append("g")
    const tooltip = d3
      .select("body")
      .append("div")
      .attr("class", "country-tooltip")
      .style("position", "absolute")
      .style("background", "rgba(0,0,0,0.85)")
      .style("color", "white")
      .style("padding", "8px 12px")
      .style("border-radius", "4px")
      .style("font-size", "13px")
      .style("pointer-events", "none")
      .style("z-index", "1000")
      .style("border", "1px solid #444")
      .style("opacity", 0)
      .style("display", "none")
    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([1, 8])
      .on("zoom", (event) => {
        g.attr("transform", event.transform.toString())
        setZoomPct(Math.round(event.transform.k * 100))
      })
    svg.call(zoom as any)
    container.addEventListener("wheel", (e) => e.preventDefault(), { passive: false })
    zoomApi.current = {
      in: () => svg.transition().duration(300).call(zoom.scaleBy as any, 1.5),
      out: () => svg.transition().duration(300).call(zoom.scaleBy as any, 0.67),
      reset: () => svg.transition().duration(500).call(zoom.transform as any, d3.zoomIdentity),
    }
    let geo: any = null
    let projection: d3.GeoProjection | null = null
    let path: d3.GeoPath | null = null
    let countries: d3.Selection<any, any, any, any> | null = null
    const fit = () => {
      if (!geo || !container) return
      const w = container.clientWidth
      const h = container.clientHeight
      const pad = 20
      projection = d3.geoNaturalEarth1().fitExtent([[pad, pad], [w - pad, h - pad]], geo)
      path = d3.geoPath().projection(projection)
      svg.attr("viewBox", `0 0 ${w} ${h}`).attr("preserveAspectRatio", "xMidYMid meet")
      countries?.attr("d", path as any)
    }
    const applyUpdate = (countryData: Record<string, number>, animate: boolean) => {
      if (!countries) return
      const vals = Object.values(countryData)
      const max = vals.length ? Math.max(...vals) : 1
      setMaxVisitors(max)
      const flash = new Set<string>()
      if (prevData) {
        for (const [code, count] of Object.entries(countryData)) {
          if (count > (prevData[code] || 0)) flash.add(code)
        }
      }
      const fillFn = (d: any) => {
        const cc = codeOf(d)
        return getCountryColor(cc ? countryData[cc] || 0 : 0, max)
      }
      ;(animate ? countries.transition().duration(500) : countries.interrupt()).attr("fill", fillFn as any)
      if (flash.size) {
        countries
          .filter((d: any) => flash.has(codeOf(d) as string))
          .interrupt()
          .attr("fill", fillFn as any)
          .transition().duration(500).attr("fill", FLASH_COLOR)
          .transition().duration(500).attr("fill", fillFn as any)
      }
      prevData = { ...countryData }
    }
    updateRef.current = applyUpdate
    ;(async () => {
      const world: any = await d3.json("https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json")
      if (!mounted) return
      geo = topojson.feature(world, world.objects.countries)
      fit()
      countries = g
        .selectAll("path")
        .data(geo.features)
        .enter()
        .append("path")
        .attr("class", "country")
        .attr("d", path as any)
        .attr("fill", NO_VISITORS)
        .attr("stroke", "#2d2d44")
        .attr("stroke-width", "0.5px")
        .style("vector-effect", "non-scaling-stroke")
        .on("mouseover", function (_e: any, d: any) {
          const cc = codeOf(d)
          const visitors = cc ? dataRef.current[cc] || 0 : 0
          tooltip.style("display", "block").transition().duration(200).style("opacity", 1)
          tooltip.html(
            `<div style="font-weight:bold;margin-bottom:4px">${nameOf(d)} (${cc || "??"})</div>` +
              `<div style="color:#4ecdc4">${visitors.toLocaleString()} visitor${visitors !== 1 ? "s" : ""}</div>`
          )
        })
        .on("mousemove", (e: any) => tooltip.style("left", e.pageX + 15 + "px").style("top", e.pageY - 10 + "px"))
        .on("mouseout", function () {
          tooltip.transition().duration(200).style("opacity", 0).on("end", () => tooltip.style("display", "none"))
        })
      applyUpdate(dataRef.current, false)
      firstApplied.current = true
    })()
    const ro = new ResizeObserver(() => {
      fit()
      zoomApi.current?.reset()
    })
    ro.observe(container)
    return () => {
      mounted = false
      ro.disconnect()
      tooltip.remove()
      d3.select(container).select("svg").remove()
    }
  }, [])
  useEffect(() => {
    if (firstApplied.current && updateRef.current) updateRef.current(data, true)
  }, [data])
  return (
    <div className="relative w-full h-full rounded-md overflow-hidden bg-[#1a1a2e]">
      <div ref={containerRef} className="w-full h-full cursor-grab active:cursor-grabbing" />
      <div className="absolute top-2 right-2 flex flex-col gap-1 z-10">
        <button className="w-8 h-8 rounded bg-black/70 border border-neutral-700 text-white flex items-center justify-center hover:bg-neutral-700" onClick={() => zoomApi.current?.in()}><Plus className="h-4 w-4" /></button>
        <button className="w-8 h-8 rounded bg-black/70 border border-neutral-700 text-white flex items-center justify-center hover:bg-neutral-700" onClick={() => zoomApi.current?.out()}><Minus className="h-4 w-4" /></button>
        <button className="w-8 h-8 rounded bg-black/70 border border-neutral-700 text-white flex items-center justify-center hover:bg-neutral-700" onClick={() => zoomApi.current?.reset()}><Home className="h-4 w-4" /></button>
      </div>
      <div className="absolute bottom-3 right-3 bg-black/70 rounded px-3 py-2 text-[11px] z-10">
        <div className="text-neutral-300 text-[10px] mb-1 text-center">Visitors</div>
        <div
          className="w-[150px] h-3 rounded-sm mb-1"
          style={{ background: `linear-gradient(to right, ${NO_VISITORS} 0%, ${MIN_COLOR} 1%, ${MID_COLOR} 50%, ${MAX_COLOR} 100%)` }}
        />
        <div className="flex justify-between text-neutral-400">
          <span>0</span>
          <span>{maxVisitors > 1 ? Math.round(Math.sqrt(maxVisitors)).toLocaleString() : "-"}</span>
          <span>{maxVisitors.toLocaleString()}</span>
        </div>
      </div>
      <div className="absolute bottom-3 left-3 bg-black/70 rounded px-2 py-1 text-[11px] text-neutral-400 z-10">
        {zoomPct}% · Scroll to zoom · Drag to pan
      </div>
    </div>
  )
}