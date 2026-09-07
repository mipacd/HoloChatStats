import { useMemo, useState, useCallback, useRef, useEffect } from "react"
import DeckGL from "@deck.gl/react"
import { OrthographicView, LinearInterpolator } from "@deck.gl/core"
import { ScatterplotLayer, TextLayer, LineLayer } from "@deck.gl/layers"
import { UserSearchDialog } from "./UserSearchDialog"
import * as d3 from "d3"
/* ── Types ─────────────────────────────────────────────────────────────── */
export interface CommunityGraphData {
  title: string
  channels: ChannelNode[]
  users: {
    count: number
    x: number[]
    y: number[]
    community: number[]
    degree: number[]
  }
  edges?: {
    source: number[]
    target: number[]
    weight: number[]
  }
  stats: {
    user_count: number
    channel_count: number
    edge_count: number
    community_count: number
  }
}
export interface ChannelNode {
  name: string
  x: number
  y: number
  community: number
  degree: number
}
/** Pixel-space footprint of a node, measured outwards from its anchor point. */
interface FitMargins {
  left: number
  right: number
  top: number
  bottom: number
}
interface FitNode extends FitMargins {
  x: number
  y: number
}
interface ChannelDatum {
  index: number
  position: [number, number, number]
  radius: number
  haloRadius: number
  ringRadius: number
  fillColor: [number, number, number, number]
  haloColor: [number, number, number, number]
  labelOffset: [number, number]
  fit: FitMargins
  name: string
  community: number
  degree: number
}
interface UserEdge {
  channelIndex: number
  weight: number
  channelName: string
}
/** Two mutually exclusive highlight modes. */
type Selection =
  | {
      kind: "user"
      userIndex: number
      userPos: [number, number, number]
      edges: UserEdge[]
      label?: string            // populated when the user came from the search dialog
    }
  | { kind: "channel"; channelIndex: number }
interface Props {
  data: CommunityGraphData
  width: number
  height: number
  month: string
  channelGroup: string
}
/* ── Palette ───────────────────────────────────────────────────────────── */
const COMMUNITY_COLORS: [number, number, number][] = [
  [56, 166, 247],
  [255, 99, 71],
  [50, 205, 100],
  [255, 193, 37],
  [180, 100, 255],
  [0, 210, 210],
  [255, 140, 66],
  [200, 80, 160],
  [120, 200, 80],
  [100, 149, 237],
]
function buildPalette(n: number): [number, number, number][] {
  if (n <= 0) return []
  if (n <= COMMUNITY_COLORS.length) return COMMUNITY_COLORS.slice(0, n)
  return [
    ...COMMUNITY_COLORS,
    ...Array.from({ length: n - COMMUNITY_COLORS.length }, (_, i) => {
      const hue = ((COMMUNITY_COLORS.length + i) * 137.508) % 360
      const rgb = d3.rgb(d3.hsl(hue, 0.82, 0.58))
      return [Math.round(rgb.r), Math.round(rgb.g), Math.round(rgb.b)] as [number, number, number]
    }),
  ]
}
/** Push a colour toward white so a line reads on top of its own halo. */
function lighten(c: readonly number[], t: number): [number, number, number] {
  return [
    Math.round(c[0] + (255 - c[0]) * t),
    Math.round(c[1] + (255 - c[1]) * t),
    Math.round(c[2] + (255 - c[2]) * t),
  ]
}
/* ── Camera constants ──────────────────────────────────────────────────── */
const MIN_ZOOM = -3
const MAX_ZOOM = 14
const MAX_FIT_ZOOM = 5
const TRANSITION_MS = 600
const VIEW_TRANSITION = new LinearInterpolator({ transitionProps: ["target", "zoom"] })
const easeInOutCubic = (t: number) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2
/* ── Pixel footprints ──────────────────────────────────────────────────── */
const LABEL_FONT_PX = 13
const LABEL_CHAR_PX = LABEL_FONT_PX * 0.58
const LABEL_MAX_PX = 170
const USER_MARGIN = 14
const USER_DOT_MARGIN: FitMargins = { left: 6, right: 6, top: 6, bottom: 6 }
function channelMargins(radius: number, name: string): FitMargins {
  const labelW = Math.min(name.length * LABEL_CHAR_PX + 10, LABEL_MAX_PX)
  return {
    left: radius + 6,
    right: radius + 7 + labelW,
    top: Math.max(radius + 6, 16),
    bottom: Math.max(radius + 6, 16),
  }
}
/* ── Channel focus ─────────────────────────────────────────────────────── */
/** Frame this fraction of a channel's audience (ignore far-flung outliers). */
const CHANNEL_FOCUS_PCTL = 0.8
const CHANNEL_FOCUS_MIN_R = 14
const AUDIENCE_SAMPLE = 20000
/* ── View fitting ──────────────────────────────────────────────────────── */
type Padding = FitMargins
interface CameraFit {
  target: [number, number, number]
  zoom: number
}
const PANEL_W = 288
const PANEL_GAP = 16
function fitAllPadding(width: number, height: number): Padding {
  const b = Math.max(20, Math.min(50, Math.min(width, height) * 0.06))
  return { top: b, right: b, bottom: b, left: b }
}
function selectionPadding(width: number, height: number): Padding {
  const b = Math.max(24, Math.min(60, Math.min(width, height) * 0.08))
  const panel = width >= 640 ? PANEL_W + PANEL_GAP * 2 : 0
  return { top: b, right: b + panel, bottom: b, left: b }
}
function clampPadding(p: Padding, width: number, height: number): Padding {
  let { left, right, top, bottom } = p
  const minW = Math.max(width * 0.35, 40)
  const minH = Math.max(height * 0.35, 40)
  if (width - left - right < minW) {
    const k = Math.max(width - minW, 0) / Math.max(left + right, 1)
    left *= k
    right *= k
  }
  if (height - top - bottom < minH) {
    const k = Math.max(height - minH, 0) / Math.max(top + bottom, 1)
    top *= k
    bottom *= k
  }
  return { left, right, top, bottom }
}
function requiredExtent(nodes: FitNode[], s: number, center?: [number, number]) {
  if (center) {
    let halfW = 0
    let halfH = 0
    for (const n of nodes) {
      const dx = (n.x - center[0]) * s
      const dy = (n.y - center[1]) * s
      halfW = Math.max(halfW, dx + n.right, -dx + n.left)
      halfH = Math.max(halfH, dy + n.bottom, -dy + n.top)
    }
    return { width: halfW * 2, height: halfH * 2, cx: center[0], cy: center[1] }
  }
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity
  for (const n of nodes) {
    xMin = Math.min(xMin, n.x * s - n.left)
    xMax = Math.max(xMax, n.x * s + n.right)
    yMin = Math.min(yMin, n.y * s - n.top)
    yMax = Math.max(yMax, n.y * s + n.bottom)
  }
  return {
    width: xMax - xMin,
    height: yMax - yMin,
    cx: (xMin + xMax) / 2 / s,
    cy: (yMin + yMax) / 2 / s,
  }
}
function fitView(
  nodes: FitNode[],
  width: number,
  height: number,
  rawPadding: Padding,
  center?: [number, number],
): CameraFit | null {
  if (!nodes.length || width <= 0 || height <= 0) return null
  const padding = clampPadding(rawPadding, width, height)
  const availW = Math.max(width - padding.left - padding.right, 1)
  const availH = Math.max(height - padding.top - padding.bottom, 1)
  const capX = availW * 0.4
  const capY = availH * 0.4
  const safe: FitNode[] = nodes.map((n) => ({
    x: n.x,
    y: n.y,
    left: Math.min(n.left, capX),
    right: Math.min(n.right, capX),
    top: Math.min(n.top, capY),
    bottom: Math.min(n.bottom, capY),
  }))
  const fits = (s: number) => {
    const e = requiredExtent(safe, s, center)
    return e.width <= availW && e.height <= availH
  }
  let lo = Math.pow(2, MIN_ZOOM)
  let hi = Math.pow(2, MAX_FIT_ZOOM)
  let scale: number
  if (fits(hi)) scale = hi
  else if (!fits(lo)) scale = lo
  else {
    for (let i = 0; i < 36; i++) {
      const mid = Math.sqrt(lo * hi)
      if (fits(mid)) lo = mid
      else hi = mid
    }
    scale = lo
  }
  const { cx, cy } = requiredExtent(safe, scale, center)
  const dxPx = (padding.left - padding.right) / 2
  const dyPx = (padding.top - padding.bottom) / 2
  return { target: [cx - dxPx / scale, cy - dyPx / scale, 0], zoom: Math.log2(scale) }
}
/* ── Connection line styling ───────────────────────────────────────────── */
const LINE_CASING: [number, number, number, number] = [6, 8, 18, 240]
const LINE_CASING_EXTRA_PX = 4
type RGBA = [number, number, number, number]
/** Background user-dot opacity, per mode. Channel mode needs far more
 *  separation because *both* groups are dots — in user mode the signal is a
 *  single highlighted dot plus the connection lines. */
const DIM_USER_OPACITY: Record<"user" | "channel", number> = {
  user: 0.1,
  channel: 0.045,
}
/** In channel mode the community colour is noise — membership is the signal. */
const DIM_USER_COLOR: RGBA = [118, 124, 140, 255]

/** Alpha levels applied to *non-selected* channels, per mode. */
interface DimStyle { fill: number; ring: number; label: number; labelBg: number }
const DIM_STYLE: Record<"user" | "channel", DimStyle> = {
  // User mode: the two or three connected channels are the whole story.
  user:    { fill: 50, ring: 40, label: 70,  labelBg: 60 },
  // Channel mode: you're reading the audience spill-over, so the *other*
  // channel names have to stay legible — they're the frame of reference.
  channel: { fill: 80, ring: 90, label: 180, labelBg: 195 },
}
const ACTIVE_RING: RGBA = [255, 255, 255, 255]
const ACTIVE_LABEL: RGBA = [255, 255, 255, 255]
const ACTIVE_LABEL_BG: RGBA = [15, 15, 30, 220]


/* ── Component ─────────────────────────────────────────────────────────── */
export function CommunityGraphView({ data, width, height, month, channelGroup }: Props) {
  const { channels, users, stats } = data
  const [selection, setSelection] = useState<Selection | null>(null)
  /* ── Palette ─────────────────────────────────────────────────────────── */
  const palette = useMemo(
    () => buildPalette(stats.community_count),
    [stats.community_count],
  )
  
  /* ── Channel data (rank-based radius + pixel footprint) ──────────────── */
  const channelData: ChannelDatum[] = useMemo(() => {
    const MIN_R = 6,
      MAX_R = 50
    const n = channels.length
    const sorted = [...channels].sort((a, b) => a.degree - b.degree)
    const rankOf = new Map<string, number>()
    sorted.forEach((ch, i) => rankOf.set(ch.name, i))
    return channels.map((ch, idx) => {
      const rank = rankOf.get(ch.name) ?? 0
      const t = n > 1 ? rank / (n - 1) : 0.5
      const r = MIN_R + t * (MAX_R - MIN_R)
      const [cr, cg, cb] = palette[ch.community] ?? [128, 128, 128]
      return {
        index: idx,
        position: [ch.x, ch.y, 0] as [number, number, number],
        radius: r,
        haloRadius: r + 4,
        ringRadius: r + 2,
        fillColor: [cr, cg, cb, 255] as [number, number, number, number],
        haloColor: [cr, cg, cb, 100] as [number, number, number, number],
        labelOffset: [r + 7, 0] as [number, number],
        fit: channelMargins(r, ch.name),
        name: ch.name,
        community: ch.community,
        degree: ch.degree,
      }
    })
  }, [channels, palette])
  const channelDataRef = useRef(channelData)
  channelDataRef.current = channelData
  /* ── User edge index (user → channels) ───────────────────────────────── */
  const userEdgeIndex = useMemo(() => {
    if (!data.edges) return null
    const map = new Map<number, { channelIndex: number; weight: number }[]>()
    const { source, target, weight } = data.edges
    for (let i = 0; i < source.length; i++) {
      const uid = source[i]
      if (!map.has(uid)) map.set(uid, [])
      map.get(uid)!.push({ channelIndex: target[i], weight: weight[i] })
    }
    return map
  }, [data.edges])
  /* ── User binary buffers ─────────────────────────────────────────────── */
  const userBuffers = useMemo(() => {
    const n = users.count
    const positions = new Float32Array(n * 3)
    const colors = new Uint8Array(n * 4)
    for (let i = 0; i < n; i++) {
      const i3 = i * 3
      positions[i3] = users.x[i]
      positions[i3 + 1] = users.y[i]
      const [r, g, b] = palette[users.community[i]] ?? [128, 128, 128]
      const i4 = i * 4
      colors[i4] = r
      colors[i4 + 1] = g
      colors[i4 + 2] = b
      colors[i4 + 3] = 160
    }
    return { length: n, positions, colors }
  }, [users, palette])
  const channelMode = selection?.kind === "channel"
  const userLayerData = useMemo(() => {
    const attributes: Record<string, any> = {
      getPosition: { value: userBuffers.positions, size: 3 },
    }
    // In channel mode every background dot is flat grey → don't upload colours.
    if (!channelMode) {
      attributes.getFillColor = { value: userBuffers.colors, size: 4 }
    }
    return { length: userBuffers.length, attributes }
  }, [userBuffers, channelMode])
  /* ── Channel mode: audience, shared channels, focus radius ───────────── */
  const channelAudience = useMemo(() => {
    if (selection?.kind !== "channel") return null
    const ci = selection.channelIndex
    const ch = channelData[ci]
    if (!ch || !data.edges) return null
    const { source, target } = data.edges
    const nEdges = source.length
    // 1 ▪ membership
    const isMember = new Uint8Array(users.count)
    let count = 0
    for (let i = 0; i < nEdges; i++) {
      if (target[i] !== ci) continue
      const u = source[i]
      if (!isMember[u]) {
        isMember[u] = 1
        count++
      }
    }
    // 2 ▪ packed buffers for the highlight layer
    const memberIds = new Uint32Array(count)
    const positions = new Float32Array(count * 3)
    const colors = new Uint8Array(count * 4)
    let k = 0
    for (let u = 0; u < users.count; u++) {
      if (!isMember[u]) continue
      memberIds[k] = u
      positions[k * 3] = users.x[u]
      positions[k * 3 + 1] = users.y[u]
      const [r, g, b] = palette[users.community[u]] ?? [128, 128, 128]
      colors[k * 4] = r
      colors[k * 4 + 1] = g
      colors[k * 4 + 2] = b
      colors[k * 4 + 3] = 235
      k++
    }
    // 3 ▪ shared audience with every other channel
    const co = new Float64Array(channels.length)
    for (let i = 0; i < nEdges; i++) {
      if (isMember[source[i]]) co[target[i]]++
    }
    const shared = Array.from(co, (v, idx) => ({ idx, v }))
      .filter((d) => d.idx !== ci && d.v > 0)
      .sort((a, b) => b.v - a.v)
      .slice(0, 5)
      .map((d) => ({
        name: channels[d.idx].name,
        users: d.v,
        pct: count ? d.v / count : 0,
      }))
    // 4 ▪ robust orbit radius (sampled percentile) for the camera
    const step = Math.max(1, Math.floor(count / AUDIENCE_SAMPLE))
    const dists: number[] = []
    for (let i = 0; i < count; i += step) {
      const u = memberIds[i]
      dists.push(Math.hypot(users.x[u] - ch.position[0], users.y[u] - ch.position[1]))
    }
    dists.sort((a, b) => a - b)
    const q = dists.length
      ? dists[Math.min(dists.length - 1, Math.floor(dists.length * CHANNEL_FOCUS_PCTL))]
      : 0
    const focusRadius = Math.max(CHANNEL_FOCUS_MIN_R, q)
    return { channelIndex: ci, count, memberIds, positions, colors, shared, focusRadius }
  }, [selection, data.edges, users, channels, channelData, palette])
  /* ── View state ──────────────────────────────────────────────────────── */
  const fitAllVS = useMemo(() => {
    const nodes: FitNode[] = channelData.map((c) => ({
      x: c.position[0],
      y: c.position[1],
      ...c.fit,
    }))
    const fit = fitView(nodes, width, height, fitAllPadding(width, height))
    return {
      target: fit?.target ?? ([0, 0, 0] as [number, number, number]),
      zoom: fit?.zoom ?? 0,
      minZoom: MIN_ZOOM,
      maxZoom: MAX_ZOOM,
    }
  }, [channelData, width, height])
  const [viewState, setViewState] = useState<Record<string, any>>(fitAllVS)
  const didFitRef = useRef(false)
  useEffect(() => {
    if (didFitRef.current || width <= 0 || height <= 0) return
    didFitRef.current = true
    setViewState(fitAllVS)
  }, [fitAllVS, width, height])
  const onViewStateChange = useCallback(
    ({ viewState: vs }: any) => setViewState(vs),
    [],
  )
  const flyTo = useCallback((fit: CameraFit, duration = TRANSITION_MS) => {
    setViewState((prev) => ({
      ...prev,
      ...fit,
      transitionDuration: duration,
      transitionEasing: easeInOutCubic,
      transitionInterpolator: VIEW_TRANSITION,
    }))
  }, [])
  const resetView = useCallback(() => {
    setSelection(null)
    flyTo({ target: fitAllVS.target, zoom: fitAllVS.zoom })
  }, [fitAllVS, flyTo])
  /* ── Focus the selection ─────────────────────────────────────────────
   * user    → centre the user, fit all of its channels
   * channel → centre the channel, fit its label + its loyal orbit
   * null    → leave the camera exactly where it is
   * ------------------------------------------------------------------- */
  const focusedRef = useRef<string | null>(null)
  useEffect(() => {
    if (!selection) {
      focusedRef.current = null
      return
    }
    let fit: CameraFit | null = null
    let key: string
    if (selection.kind === "user") {
      key = `user:${selection.userIndex}`
      const center: [number, number] = [selection.userPos[0], selection.userPos[1]]
      const nodes: FitNode[] = [
        { x: center[0], y: center[1], left: USER_MARGIN, right: USER_MARGIN, top: USER_MARGIN, bottom: USER_MARGIN },
      ]
      for (const e of selection.edges) {
        const ch = channelDataRef.current[e.channelIndex]
        if (ch) nodes.push({ x: ch.position[0], y: ch.position[1], ...ch.fit })
      }
      fit = fitView(nodes, width, height, selectionPadding(width, height), center)
    } else {
      key = `channel:${selection.channelIndex}`
      const ch = channelDataRef.current[selection.channelIndex]
      if (!ch) return
      const [cx, cy] = ch.position
      const R = channelAudience?.focusRadius ?? CHANNEL_FOCUS_MIN_R
      const nodes: FitNode[] = [
        { x: cx, y: cy, ...ch.fit },
        { x: cx + R, y: cy, ...USER_DOT_MARGIN },
        { x: cx - R, y: cy, ...USER_DOT_MARGIN },
        { x: cx, y: cy + R, ...USER_DOT_MARGIN },
        { x: cx, y: cy - R, ...USER_DOT_MARGIN },
      ]
      fit = fitView(nodes, width, height, selectionPadding(width, height), [cx, cy])
    }
    if (!fit) return
    const isNewFocus = focusedRef.current !== key
    focusedRef.current = key
    flyTo(fit, isNewFocus ? TRANSITION_MS : 0)
  }, [selection, channelAudience, width, height, flyTo])
  /* ── Selection-derived data ──────────────────────────────────────────── */
  const activeChannels = useMemo(() => {
    if (!selection) return null
    if (selection.kind === "user") return new Set(selection.edges.map((e) => e.channelIndex))
    return new Set([selection.channelIndex])
  }, [selection])
  const displayChannelData = useMemo(() => {
    if (!activeChannels || !selection) {
      return channelData.map((ch) => ({
        ...ch,
        ringColor: ACTIVE_RING,
        labelColor: ACTIVE_LABEL,
        labelBgColor: ACTIVE_LABEL_BG,
      }))
    }
    const dim = DIM_STYLE[selection.kind]
    return channelData.map((ch) =>
      activeChannels.has(ch.index)
        ? {
            ...ch,
            ringColor: ACTIVE_RING,
            labelColor: ACTIVE_LABEL,
            labelBgColor: ACTIVE_LABEL_BG,
          }
        : {
            ...ch,
            fillColor: [ch.fillColor[0], ch.fillColor[1], ch.fillColor[2], dim.fill] as RGBA,
            haloColor: [0, 0, 0, 0] as RGBA,
            ringColor: [255, 255, 255, dim.ring] as RGBA,
            labelColor: [255, 255, 255, dim.label] as RGBA,
            labelBgColor: [15, 15, 30, dim.labelBg] as RGBA,
          },
    )
  }, [channelData, activeChannels, selection])
  const connectionLines = useMemo(() => {
    if (selection?.kind !== "user") return []
    const maxW = Math.max(...selection.edges.map((e) => e.weight), 1)
    return selection.edges.map((e) => {
      const ch = channelData[e.channelIndex]
      const base = ch ? ch.fillColor.slice(0, 3) : [255, 255, 255]
      const core = lighten(base, 0.45)
      const w = 1.5 + (e.weight / maxW) * 5
      return {
        sourcePosition: selection.userPos,
        targetPosition: ch?.position ?? ([0, 0, 0] as [number, number, number]),
        width: w,
        casingWidth: w + LINE_CASING_EXTRA_PX,
        color: [core[0], core[1], core[2], 255] as [number, number, number, number],
      }
    })
  }, [selection, channelData])
  /* ── Click handling ──────────────────────────────────────────────────── */
  const selectUser = useCallback(
    (idx: number, label?: string) => {
      if (!userEdgeIndex) return
      if (selection?.kind === "user" && selection.userIndex === idx) return
      const edges = userEdgeIndex.get(idx)
      if (!edges?.length) return
      setSelection({
        kind: "user",
        userIndex: idx,
        userPos: [users.x[idx], users.y[idx], 0],
        label,
        edges: edges
          .map((e) => ({ ...e, channelName: channels[e.channelIndex]?.name ?? "Unknown" }))
          .sort((a, b) => b.weight - a.weight),
      })
    },
    [userEdgeIndex, users, channels, selection],
  )
  /** Called by the search dialog. Returns an error message, or null on success. */
  const focusUserByIndex = useCallback(
    (idx: number, username: string): string | null => {
      if (!Number.isInteger(idx) || idx < 0 || idx >= users.count) {
        return "That user isn't in the loaded graph — reload it and try again."
      }
      if (!userEdgeIndex?.get(idx)?.length) {
        return "That user has no channel activity in the loaded graph."
      }
      selectUser(idx, username)
      return null
    },
    [users.count, userEdgeIndex, selectUser],
  )
  const handleClick = useCallback(
    (info: any) => {
      const id = info.layer?.id
      // Channel circle → channel highlight mode
      if (id === "channels" && info.object) {
        const ci = info.object.index as number
        if (selection?.kind === "channel" && selection.channelIndex === ci) return
        setSelection({ kind: "channel", channelIndex: ci })
        return
      }
      // Any user dot (dimmed base layer or highlighted audience) → user mode
      if (id === "channel-users" && info.index >= 0 && channelAudience) {
        selectUser(channelAudience.memberIds[info.index])
        return
      }
      if (id === "users" && info.index >= 0) {
        selectUser(info.index)
        return
      }
      // Empty space → deselect, camera untouched
      setSelection(null)
    },
    [selection, channelAudience, selectUser],
  )
  /* ── Layers ──────────────────────────────────────────────────────────── */
  const layers = useMemo(() => {
    const dimmed = !!selection
    const userMode = selection?.kind === "user"
    return [
      /* 1 ▪ All user dots (dimmed whenever anything is selected) */
      new ScatterplotLayer({
        id: "users",
        data: userLayerData as any,
        // ignored when the binary getFillColor attribute is present
        getFillColor: DIM_USER_COLOR,
        radiusUnits: "common" as const,
        getRadius: 0.35,
        radiusMinPixels: 0.5,
        radiusMaxPixels: dimmed ? 2.5 : 4,
        opacity: selection ? DIM_USER_OPACITY[selection.kind] : 1.0,
        pickable: !!userEdgeIndex,
        autoHighlight: false,
      }),
      /* 2 ▪ Channel mode: the channel's audience, full brightness on top */
      ...(channelAudience && channelAudience.count
        ? [
            new ScatterplotLayer({
              id: "channel-users",
              data: {
                length: channelAudience.count,
                attributes: {
                  getPosition: { value: channelAudience.positions, size: 3 },
                  getFillColor: { value: channelAudience.colors, size: 4 },
                },
              } as any,
              radiusUnits: "common" as const,
              getRadius: 0.5,
              radiusMinPixels: 1.6,
              radiusMaxPixels: 6,
              opacity: 1,
              pickable: !!userEdgeIndex,
              autoHighlight: false,
            }),
          ]
        : []),
      /* 3 ▪ User mode: connection casing + brightened core */
      ...(connectionLines.length
        ? [
            new LineLayer({
              id: "connection-casings",
              data: connectionLines,
              getSourcePosition: (d: any) => d.sourcePosition,
              getTargetPosition: (d: any) => d.targetPosition,
              getColor: () => LINE_CASING,
              getWidth: (d: any) => d.casingWidth,
              widthUnits: "pixels" as const,
              widthMinPixels: 5,
              pickable: false,
            }),
            new LineLayer({
              id: "connections",
              data: connectionLines,
              getSourcePosition: (d: any) => d.sourcePosition,
              getTargetPosition: (d: any) => d.targetPosition,
              getColor: (d: any) => d.color,
              getWidth: (d: any) => d.width,
              widthUnits: "pixels" as const,
              widthMinPixels: 1.5,
              pickable: false,
            }),
          ]
        : []),
      /* 4 ▪ Channel halo */
      new ScatterplotLayer({
        id: "channel-halos",
        data: displayChannelData,
        getPosition: (d: any) => d.position,
        getFillColor: (d: any) => d.haloColor,
        getRadius: (d: any) => d.haloRadius,
        radiusUnits: "pixels" as const,
        stroked: false,
        pickable: false,
        updateTriggers: { getFillColor: [activeChannels] },
      }),
      /* 5 ▪ White border ring */
      new ScatterplotLayer({
        id: "channel-rings",
        data: displayChannelData,
        getPosition: (d: any) => d.position,
        getFillColor: (d: any) => d.ringColor,
        getRadius: (d: any) => d.ringRadius,
        radiusUnits: "pixels" as const,
        stroked: false,
        pickable: false,
        updateTriggers: { getFillColor: [displayChannelData] },
      }),
      /* 6 ▪ Channel fill — always pickable, so you can hop channel → channel */
      new ScatterplotLayer({
        id: "channels",
        data: displayChannelData,
        getPosition: (d: any) => d.position,
        getFillColor: (d: any) => d.fillColor,
        getRadius: (d: any) => d.radius,
        radiusUnits: "pixels" as const,
        stroked: false,
        pickable: true,
        updateTriggers: { getFillColor: [activeChannels] },
      }),
      /* 7 ▪ Labels */
      new TextLayer({
        id: "labels",
        data: displayChannelData,
        getPosition: (d: any) => d.position,
        getText: (d: any) => d.name,
        getSize: LABEL_FONT_PX,
        getColor: (d: any) => d.labelColor,
        getTextAnchor: "start",
        getAlignmentBaseline: "center",
        getPixelOffset: (d: any) => d.labelOffset,
        fontFamily: "Inter, system-ui, sans-serif",
        fontWeight: "bold",
        background: true,
        getBackgroundColor: (d: any) => d.labelBgColor,
        backgroundPadding: [4, 6],
        outlineWidth: 0,
        updateTriggers: {
          getColor: [displayChannelData],
          getBackgroundColor: [displayChannelData],
        },
      }),
      /* 8 ▪ Selected-user highlight dot */
      ...(userMode && selection?.kind === "user"
        ? [
            new ScatterplotLayer({
              id: "user-highlight",
              data: [{ position: selection.userPos }],
              getPosition: (d: any) => d.position,
              getFillColor: [255, 255, 255, 255],
              getLineColor: [255, 220, 50, 255],
              lineWidthMinPixels: 3,
              stroked: true,
              radiusUnits: "pixels" as const,
              getRadius: 8,
              pickable: false,
            }),
          ]
        : []),
    ]
  }, [
    userLayerData,
    channelAudience,
    displayChannelData,
    activeChannels,
    connectionLines,
    selection,
    userEdgeIndex,
  ])
  /* ── Render ──────────────────────────────────────────────────────────── */
  const selectedChannel =
    selection?.kind === "channel" ? channelData[selection.channelIndex] : null
  return (
    <div className="relative w-full h-full">
      <DeckGL
        width={width}
        height={height}
        views={new OrthographicView()}
        viewState={viewState}
        onViewStateChange={onViewStateChange}
        controller={true}
        layers={layers}
        onClick={handleClick}
        getCursor={({ isHovering }: any) => (isHovering ? "pointer" : "grab")}
        getTooltip={({ object, layer }: any) =>
          // tooltips stay available in channel mode so you can hop around
          selection?.kind !== "user" && layer?.id === "channels" && object
            ? {
                html: `<b>${object.name}</b><br/>Community ${object.community}<br/>${object.degree.toLocaleString()} users`,
                style: {
                  backgroundColor: "rgba(15,15,30,0.95)",
                  color: "#fff",
                  padding: "8px 12px",
                  borderRadius: "6px",
                  fontSize: "13px",
                  border: "1px solid rgba(255,255,255,0.15)",
                },
              }
            : null
        }
      />
      <div className="absolute top-4 left-4 z-20">
        <UserSearchDialog
          month={month}
          channelGroup={channelGroup}
          onFound={focusUserByIndex}
        />
      </div>
      <button
        onClick={resetView}
        className="absolute bottom-4 left-4 z-20 rounded-md border border-white/15 bg-zinc-900/90 px-3 py-1.5 text-xs text-zinc-300 shadow-lg backdrop-blur hover:bg-white/10 hover:text-white"
      >
        Reset view
      </button>
      {/* ── User mode panel ─────────────────────────────────────────── */}
      {selection?.kind === "user" && (
        <div className="absolute top-4 right-4 z-20 w-72 rounded-lg border border-white/15 bg-zinc-900/95 p-4 text-sm text-white shadow-xl backdrop-blur">
          <div className="mb-3 flex items-center justify-between">
            <span className="truncate text-base font-semibold">
              {selection.label ?? "User Activity"}
            </span>
            <button
              className="rounded px-2 py-0.5 text-xs text-zinc-400 hover:bg-white/10 hover:text-white"
              onClick={() => setSelection(null)}
            >
              ✕
            </button>
          </div>
          <table className="w-full">
            <thead>
              <tr className="text-xs text-zinc-400">
                <th className="pb-1 text-left font-medium">Channel</th>
                <th className="pb-1 text-right font-medium">Messages</th>
              </tr>
            </thead>
            <tbody>
              {selection.edges.slice(0, 5).map((edge, i) => (
                <tr key={i} className="border-t border-white/10">
                  <td className="py-1.5 pr-3">{edge.channelName}</td>
                  <td className="py-1.5 text-right tabular-nums text-zinc-400">
                    {Math.round(edge.weight).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {selection.edges.length > 5 && (
            <p className="mt-1 text-xs text-zinc-500">
              + {selection.edges.length - 5} more channel
              {selection.edges.length - 5 > 1 ? "s" : ""}
            </p>
          )}
          <p className="mt-3 text-[11px] text-zinc-500">
            Click empty space to deselect
          </p>
        </div>
      )}
      {/* ── Channel mode panel ──────────────────────────────────────── */}
      {selectedChannel && (
        <div className="absolute top-4 right-4 z-20 w-72 rounded-lg border border-white/15 bg-zinc-900/95 p-4 text-sm text-white shadow-xl backdrop-blur">
          <div className="mb-1 flex items-start justify-between gap-2">
            <span className="text-base font-semibold leading-tight">
              {selectedChannel.name}
            </span>
            <button
              className="rounded px-2 py-0.5 text-xs text-zinc-400 hover:bg-white/10 hover:text-white"
              onClick={() => setSelection(null)}
            >
              ✕
            </button>
          </div>
          <div className="mb-3 flex items-center gap-2 text-xs text-zinc-400">
            <span
              className="inline-block h-2.5 w-2.5 rounded-full"
              style={{
                backgroundColor: `rgb(${selectedChannel.fillColor.slice(0, 3).join(",")})`,
              }}
            />
            Community {selectedChannel.community} ·{" "}
            {selectedChannel.degree.toLocaleString()} users
          </div>
          {channelAudience && channelAudience.shared.length > 0 && (
            <>
              <table className="w-full">
                <thead>
                  <tr className="text-xs text-zinc-400">
                    <th className="pb-1 text-left font-medium">Shared audience</th>
                    <th className="pb-1 text-right font-medium">Users</th>
                  </tr>
                </thead>
                <tbody>
                  {channelAudience.shared.map((s, i) => (
                    <tr key={i} className="border-t border-white/10">
                      <td className="py-1.5 pr-3">{s.name}</td>
                      <td className="py-1.5 text-right tabular-nums text-zinc-400">
                        {s.users.toLocaleString()}
                        <span className="ml-1 text-zinc-600">
                          {(s.pct * 100).toFixed(0)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
          <p className="mt-3 text-[11px] text-zinc-500">
            Click a dot to inspect that user · click empty space to deselect
          </p>
        </div>
      )}
    </div>
  )
}
