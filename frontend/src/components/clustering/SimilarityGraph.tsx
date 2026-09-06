import { Suspense, lazy, useEffect, useMemo, useRef } from "react"
import * as d3 from "d3"
import { Loader2 } from "lucide-react"
import ForceGraph2D from "react-force-graph-2d"
import { useElementSize } from "@/hooks/useElementSize"
const ForceGraph3DView = lazy(() => import("./ForceGraph3DView"))
export interface GraphNode {
  id: string
  community: number
  degree: number
  neighbors: string[]
  x: number
  y: number
  z?: number
}
export interface GraphLink {
  source: string
  target: string
  weight: number
}
interface Props {
  nodes: GraphNode[]
  links: GraphLink[]
  is3d: boolean
  loading?: boolean
}
export function SimilarityGraph({ nodes, links, is3d, loading }: Props) {
  const { ref: containerRef, width, height } = useElementSize<HTMLDivElement>()
  const fgRef = useRef<any>(null)
  const safeNodes = nodes ?? []                                          // ← CHANGED
  const safeLinks = links ?? []                                          // ← CHANGED
  const communityColor = useMemo(() => {
    if (!safeNodes.length) return () => "#888888"                        // ← CHANGED
    const ids = safeNodes.map((n) => n.community)                        // ← CHANGED
    const max = ids.length ? Math.max(...ids, 1) : 1
    const scale = d3.scaleSequential(d3.interpolateViridis).domain([0, max])
    return (c: number) => scale(c)
  }, [safeNodes])                                                        // ← CHANGED
  const { minWeight, spread } = useMemo(() => {
    if (safeLinks.length === 0) return { minWeight: 0, spread: 1 }       // ← CHANGED
    const weights = safeLinks.map((l) => l.weight)                       // ← CHANGED
    const min = Math.min(...weights)
    const max = Math.max(...weights)
    return { minWeight: min, spread: max - min || 1 }
  }, [safeLinks])                                                        // ← CHANGED
  const linkColor = (link: any) => {
    const norm = (link.weight - minWeight) / spread
    const opacity = 0.1 + Math.pow(norm, 1.1) * 0.9
    return `rgba(255,255,255,${opacity.toFixed(3)})`
  }
  const graphData = useMemo(
    () => ({
      nodes: safeNodes.map((n) => ({ ...n })),                           // ← CHANGED
      links: safeLinks.map((l) => ({ ...l })),                           // ← CHANGED
    }),
    [safeNodes, safeLinks],                                              // ← CHANGED
  )
  useEffect(() => {
    let tries = 0
    let id: number
    const tryFit = () => {
      if (fgRef.current?.zoomToFit) {
        fgRef.current.zoomToFit(600, 40)
      } else if (tries < 20) {
        tries++
        id = window.setTimeout(tryFit, 150)
      }
    }
    tryFit()
    return () => clearTimeout(id)
  }, [graphData, is3d, width, height])
  const resetView = () => fgRef.current?.zoomToFit?.(600, 40)
  const nodeLabel = (n: any) =>
    `<div style="font:12px sans-serif;max-width:240px;">${n.id}<br/>Connected to: ${(n.neighbors ?? []).join(", ")}</div>`
  const linkLabel = (l: any) =>
    `<div style="font:12px sans-serif;">${l.source.id ?? l.source} ↔ ${l.target.id ?? l.target}<br/>Score: ${(l.weight * 100).toFixed(2)}</div>`
  const nodeCanvasObject = (node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const r = 5
    ctx.beginPath()
    ctx.arc(node.x, node.y, r, 0, 2 * Math.PI)
    ctx.fillStyle = communityColor(node.community)
    ctx.fill()
    ctx.lineWidth = 1 / globalScale
    ctx.strokeStyle = "black"
    ctx.stroke()
    const fontSize = Math.max(10, 12 / globalScale)
    ctx.font = `${fontSize}px sans-serif`
    ctx.fillStyle = "#ffffff"
    ctx.textAlign = "left"
    ctx.textBaseline = "middle"
    ctx.fillText(node.id, node.x + r + 3 / globalScale, node.y)
  }
  return (
    <div ref={containerRef} className="relative w-full h-full" onDoubleClick={resetView}>
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center z-10 bg-background/40">
          <Loader2 className="h-10 w-10 animate-spin text-primary" />
        </div>
      )}
      {width > 0 && height > 0 && safeNodes.length > 0 && (             /* ← CHANGED */
        is3d ? (
          <Suspense
            fallback={
              <div className="flex items-center justify-center h-full">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            }
          >
            <ForceGraph3DView
              graphData={graphData}
              width={width}
              height={height}
              communityColor={communityColor}
              linkColor={linkColor}
              nodeLabel={nodeLabel}
              linkLabel={linkLabel}
              getRef={(el) => (fgRef.current = el)}
              onEngineStop={() => fgRef.current?.zoomToFit(400, 40)}
            />
          </Suspense>
        ) : (
          <ForceGraph2D
            ref={fgRef}
            width={width}
            height={height}
            graphData={graphData as any}
            backgroundColor="rgba(0,0,0,0)"
            onEngineStop={() => fgRef.current?.zoomToFit(400, 40)}
            nodeLabel={nodeLabel}
            linkLabel={linkLabel}
            linkColor={linkColor}
            linkWidth={1}
            linkHoverPrecision={6}
            nodeCanvasObject={nodeCanvasObject}
            nodePointerAreaPaint={(node: any, color: string, ctx: CanvasRenderingContext2D) => {
              ctx.fillStyle = color
              ctx.beginPath()
              ctx.arc(node.x, node.y, 7, 0, 2 * Math.PI)
              ctx.fill()
            }}
          />
        )
      )}
    </div>
  )
}