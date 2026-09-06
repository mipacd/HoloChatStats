import { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import { ArrowDown, ArrowUp, Minus, PlayCircle, Download, CheckCircle } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
import "./viewer.css"
const USD_TO_JPY = 150
const MEMBERSHIP_PRICE_USD = 4.99
const MEMBERSHIP_PRICE_JPY = 750
const MAX_GRAPH_POINTS = 200
const ratesToUSD: Record<string, number> = {
  "$": 1, "¥": 0.0067, "￥": 0.0067, "€": 1.09, "£": 1.27, "₩": 0.00075,
  "CA$": 0.74, "A$": 0.65, "₹": 0.012, "R$": 0.2, "₱": 0.018, "NT$": 0.031, "HK$": 0.13, "MX$": 0.059,
}
const ratesToJPY: Record<string, number> = {
  "$": 150, "¥": 1, "￥": 1, "€": 163, "£": 190, "₩": 0.11,
  "CA$": 111, "A$": 97, "₹": 1.8, "R$": 30, "₱": 2.7, "NT$": 4.7, "HK$": 19, "MX$": 8.8,
}
function hasJapanese(t: string) {
  return /[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF\u3400-\u4DBF]/.test(t)
}
function isWOnly(t: string) {
  if (!t) return false
  const c = t.replace(/\s/g, "")
  return c.length > 0 && /^[wWｗ]+$/.test(c)
}
function hasActualText(t?: string) {
  if (!t) return false
  if (isWOnly(t)) return true
  const noEmoji = t
    .replace(/[\uD800-\uDBFF][\uDC00-\uDFFF]/g, "")
    .replace(/[\u2600-\u27BF]/g, "")
    .replace(/[\u2300-\u23FF]/g, "")
    .replace(/[\u2B50\u2B55]/g, "")
    .replace(/[\u3030\u303D\u3297\u3299]/g, "")
    .replace(/[\u200D\uFE0F\uFE0E]/g, "")
    .replace(/[\u0023\u002A\u0030-\u0039]\uFE0F?\u20E3/g, "")
    .replace(/[\uD83C][\uDDE6-\uDDFF]/g, "")
  return noEmoji.replace(/:[a-zA-Z0-9_-]+:/g, "").trim().length > 0
}
function isJapaneseMessage(t?: string) {
  if (!t) return false
  if (isWOnly(t)) return true
  return hasJapanese(t)
}
function parseSuperChatAmount(s: string | undefined, useJPY: boolean) {
  if (!s) return 0
  const cleaned = s.replace(/[\s,]/g, "")
  const rates = useJPY ? ratesToJPY : ratesToUSD
  for (const sym of Object.keys(rates).sort((a, b) => b.length - a.length)) {
    if (cleaned.includes(sym)) {
      const amt = parseFloat(cleaned.replace(sym, "").replace(/[^\d.]/g, ""))
      if (!isNaN(amt)) return amt * rates[sym]
    }
  }
  const amt = parseFloat(cleaned.replace(/[^\d.]/g, ""))
  return isNaN(amt) ? 0 : amt
}
function extractVideoId(url: string): string | null {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
    /youtube\.com\/v\/([^&\n?#]+)/,
    /youtube\.com\/live\/([^&\n?#]+)/,
  ]
  for (const p of patterns) {
    const m = url.match(p)
    if (m) return m[1]
  }
  return null
}
function drawGraph(canvas: HTMLCanvasElement | null, values: number[], color: string) {
  if (!canvas || values.length < 2) return
  const ctx = canvas.getContext("2d")!
  const rect = canvas.parentElement!.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1
  canvas.width = rect.width * dpr
  canvas.height = rect.height * dpr
  ctx.setTransform(1, 0, 0, 1, 0, 0)
  ctx.scale(dpr, dpr)
  ctx.clearRect(0, 0, rect.width, rect.height)
  let min = Math.min(...values)
  let max = Math.max(...values)
  const range = max - min
  if (range === 0) {
    min = min * 0.9
    max = max * 1.1 || 1
  } else {
    min -= range * 0.1
    max += range * 0.1
  }
  const pad = { top: 8, bottom: 8, left: 4, right: 4 }
  const gw = rect.width - pad.left - pad.right
  const gh = rect.height - pad.top - pad.bottom
  const pts = values.map((v, i) => ({
    x: pad.left + (i / (values.length - 1)) * gw,
    y: pad.top + gh - ((v - min) / (max - min)) * gh,
  }))
  const grad = ctx.createLinearGradient(0, pad.top, 0, rect.height - pad.bottom)
  grad.addColorStop(0, color.replace(")", ", 0.3)").replace("rgb", "rgba"))
  grad.addColorStop(1, color.replace(")", ", 0.05)").replace("rgb", "rgba"))
  ctx.beginPath()
  ctx.moveTo(pts[0].x, rect.height - pad.bottom)
  pts.forEach((p) => ctx.lineTo(p.x, p.y))
  ctx.lineTo(pts[pts.length - 1].x, rect.height - pad.bottom)
  ctx.closePath()
  ctx.fillStyle = grad
  ctx.fill()
  ctx.beginPath()
  ctx.moveTo(pts[0].x, pts[0].y)
  for (let i = 1; i < pts.length; i++) {
    const prev = pts[i - 1]
    const curr = pts[i]
    const midX = (prev.x + curr.x) / 2
    ctx.quadraticCurveTo(prev.x, prev.y, midX, (prev.y + curr.y) / 2)
  }
  ctx.lineTo(pts[pts.length - 1].x, pts[pts.length - 1].y)
  ctx.strokeStyle = color
  ctx.lineWidth = 2
  ctx.lineCap = "round"
  ctx.lineJoin = "round"
  ctx.stroke()
}
interface ChatPayload {
  author?: string
  message?: string
  isMember?: boolean
  badges?: string[]
  isMembershipGiftRedemption?: boolean
  isMembershipPurchase?: boolean
  isSuperChat?: boolean
  isSuperSticker?: boolean
  superChatAmount?: string
}
const initialStats = {
  messageCount: 0,
  messagesPerMinute: 0,
  uniqueUsers: 0,
  memberCount: 0,
  memberPercentage: "0",
  jpCount: 0,
  jpPercentage: "0",
  giftCount: 0,
  purchaseCount: 0,
  revenue: "$0",
  ccv: "-",
  ccvTrend: "" as "" | "up" | "down" | "stable",
  elapsedTime: "0:00",
}
export default function Viewer() {
  const { t, i18n } = useTranslation()
  const useJPY = i18n.language === "ja"
  const [stage, setStage] = useState<"input" | "viewing">("input")
  const [url, setUrl] = useState("")
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)
  const [videoSrc, setVideoSrc] = useState("")
  const [chatSrc, setChatSrc] = useState("")
  const [stats, setStats] = useState(initialStats)
  const [conn, setConn] = useState<{ show: boolean; connected: boolean; fading: boolean }>({
    show: false,
    connected: false,
    fading: false,
  })
  // tracking refs
  const messageCount = useRef(0)
  const textCount = useRef(0)
  const timestamps = useRef<number[]>([])
  const startTime = useRef<number | null>(null)
  const users = useRef<Set<string>>(new Set())
  const members = useRef<Set<string>>(new Set())
  const jpCount = useRef(0)
  const giftCount = useRef(0)
  const purchaseCount = useRef(0)
  const revenueBase = useRef(0)
  const initialLoadComplete = useRef(false)
  const ccvHistory = useRef<number[]>([])
  const lastCCV = useRef<number | null>(null)
  const msgPerMinHistory = useRef<{ value: number; timestamp: number }[]>([])
  const currentVideoId = useRef<string | null>(null)
  const chatFrameRef = useRef<HTMLIFrameElement>(null)
  const ccvCanvasRef = useRef<HTMLCanvasElement>(null)
  const msgCanvasRef = useRef<HTMLCanvasElement>(null)
  const statsInterval = useRef<number | null>(null)
  const ccvFetchInterval = useRef<number | null>(null)
  const initialLoadTimeout = useRef<number | null>(null)
  const connTimeout = useRef<number | null>(null)
  const formatRevenue = useCallback(
    (base: number) => (useJPY ? "¥" + Math.round(base).toLocaleString() : "$" + base.toFixed(2)),
    [useJPY]
  )
  const membershipPrice = useJPY ? MEMBERSHIP_PRICE_JPY : MEMBERSHIP_PRICE_USD
  useEffect(() => {
    registerEriContext(() => ({
      page: "viewer",
      endpoint: "/api/ccv",
      parameters: { videoId: currentVideoId.current },
      description: `Viewing live stream ${currentVideoId.current ?? ""}`,
    }))
    return () => registerEriContext(null)
  }, [])
  const updateConnectionStatus = useCallback((connected: boolean) => {
    if (connTimeout.current) window.clearTimeout(connTimeout.current)
    if (connected) {
      setConn({ show: true, connected: true, fading: false })
      connTimeout.current = window.setTimeout(() => {
        setConn((c) => ({ ...c, fading: true }))
        window.setTimeout(() => setConn({ show: false, connected: true, fading: false }), 300)
      }, 3000)
    } else {
      setConn({ show: true, connected: false, fading: false })
    }
  }, [])
  const updateCCV = useCallback((count: number | null | undefined) => {
    if (count == null || isNaN(count as number)) {
      setStats((s) => ({ ...s, ccv: "-", ccvTrend: "" }))
      return
    }
    const ccv = Math.trunc(count as number)
    let trend: "up" | "down" | "stable" = "stable"
    if (lastCCV.current !== null) {
      const diff = ccv - lastCCV.current
      const threshold = Math.max(10, lastCCV.current * 0.01)
      trend = diff > threshold ? "up" : diff < -threshold ? "down" : "stable"
    }
    ccvHistory.current.push(ccv)
    if (ccvHistory.current.length > MAX_GRAPH_POINTS)
      ccvHistory.current = ccvHistory.current.slice(-MAX_GRAPH_POINTS)
    lastCCV.current = ccv
    drawGraph(ccvCanvasRef.current, ccvHistory.current, "rgb(72, 166, 167)")
    setStats((s) => ({ ...s, ccv: ccv.toLocaleString(), ccvTrend: trend }))
  }, [])
  const handleChatMessage = useCallback(
    (m: ChatPayload) => {
      if (!startTime.current) startTime.current = Date.now()
      messageCount.current++
      if (initialLoadComplete.current) {
        timestamps.current.push(Date.now())
        const cutoff = Date.now() - 5 * 60 * 1000
        timestamps.current = timestamps.current.filter((ts) => ts > cutoff)
      }
      if (m.author) {
        users.current.add(m.author)
        if (
          m.isMember ||
          (m.badges &&
            m.badges.some(
              (b) => b && (b.toLowerCase().includes("member") || b.toLowerCase().includes("sponsor"))
            ))
        ) {
          members.current.add(m.author)
        }
      }
      if (hasActualText(m.message)) {
        textCount.current++
        if (isJapaneseMessage(m.message)) jpCount.current++
      }
      if (m.isMembershipGiftRedemption) {
        giftCount.current++
        revenueBase.current += membershipPrice
      }
      if (m.isMembershipPurchase && !m.isSuperChat && !m.isSuperSticker) {
        purchaseCount.current++
        revenueBase.current += membershipPrice
      }
      if (m.isSuperChat && m.superChatAmount) {
        const amt = parseSuperChatAmount(m.superChatAmount, useJPY)
        if (amt > 0) revenueBase.current += amt
      }
    },
    [membershipPrice, useJPY]
  )
  // window message listener (userscript protocol)
  useEffect(() => {
    const onMessage = (event: MessageEvent) => {
      if (event.origin !== "https://www.youtube.com") return
      const data = event.data
      if (!data || typeof data !== "object") return
      if (data.type === "HOLOCHATSTATS_CHAT_MESSAGE") handleChatMessage(data.payload)
      else if (data.type === "HOLOCHATSTATS_SCRIPT_READY") updateConnectionStatus(true)
      else if (data.type === "HOLOCHATSTATS_CCV_UPDATE") updateCCV(data.ccv)
    }
    window.addEventListener("message", onMessage)
    return () => window.removeEventListener("message", onMessage)
  }, [handleChatMessage, updateConnectionStatus, updateCCV])
  const flushStats = useCallback(() => {
    let elapsed = "0:00"
    if (startTime.current) {
      const e = Date.now() - startTime.current
      elapsed = `${Math.floor(e / 60000)}:${String(Math.floor((e % 60000) / 1000)).padStart(2, "0")}`
    }
    let mpm = 0
    if (initialLoadComplete.current && timestamps.current.length > 0) {
      const now = Date.now()
      const recent = timestamps.current.filter((ts) => ts > now - 5 * 60 * 1000)
      if (recent.length > 0) {
        const span = Math.min(now - timestamps.current[0], 5 * 60 * 1000) / 60000
        mpm = span > 0 ? parseFloat((recent.length / span).toFixed(1)) : 0
      }
    }
    const now = Date.now()
    const last = msgPerMinHistory.current[msgPerMinHistory.current.length - 1]
    if (!last || now - last.timestamp >= 5000) {
      msgPerMinHistory.current.push({ value: mpm, timestamp: now })
      if (msgPerMinHistory.current.length > MAX_GRAPH_POINTS)
        msgPerMinHistory.current = msgPerMinHistory.current.slice(-MAX_GRAPH_POINTS)
      drawGraph(msgCanvasRef.current, msgPerMinHistory.current.map((h) => h.value), "rgb(92, 144, 210)")
    }
    const u = users.current.size
    const mem = members.current.size
    setStats((s) => ({
      ...s,
      messageCount: messageCount.current,
      messagesPerMinute: mpm,
      uniqueUsers: u,
      memberCount: mem,
      memberPercentage: u > 0 ? ((mem / u) * 100).toFixed(1) : "0",
      jpCount: jpCount.current,
      jpPercentage: textCount.current > 0 ? ((jpCount.current / textCount.current) * 100).toFixed(1) : "0",
      giftCount: giftCount.current,
      purchaseCount: purchaseCount.current,
      revenue: formatRevenue(revenueBase.current),
      elapsedTime: elapsed,
    }))
  }, [formatRevenue])
  const fetchCCV = useCallback(() => {
    const id = currentVideoId.current
    if (!id) return
    api
      .get(`/ccv/${id}`)
      .then((res) => {
        if (res.data?.ccv != null) updateCCV(res.data.ccv)
        else throw new Error()
      })
      .catch(() => {
        chatFrameRef.current?.contentWindow?.postMessage(
          { type: "HOLOCHATSTATS_REQUEST_CCV" },
          "https://www.youtube.com"
        )
      })
  }, [updateCCV])
  const loadVideo = () => {
    const trimmed = url.trim()
    if (!trimmed) {
      setError(t("Please enter a YouTube URL"))
      return
    }
    const videoId = extractVideoId(trimmed)
    if (!videoId) {
      setError(t("Invalid YouTube URL"))
      return
    }
    setError("")
    currentVideoId.current = videoId
    // reset tracking
    messageCount.current = 0
    textCount.current = 0
    timestamps.current = []
    jpCount.current = 0
    giftCount.current = 0
    purchaseCount.current = 0
    revenueBase.current = 0
    users.current.clear()
    members.current.clear()
    startTime.current = null
    initialLoadComplete.current = false
    ccvHistory.current = []
    lastCCV.current = null
    msgPerMinHistory.current = []
    setStats({ ...initialStats, revenue: formatRevenue(0) })
    updateConnectionStatus(false)
    setVideoSrc(`https://www.youtube.com/embed/${videoId}?autoplay=1`)
    setChatSrc(
      `https://www.youtube.com/live_chat?v=${videoId}&embed_domain=${window.location.hostname}&dark_theme=1`
    )
    setStage("viewing")
    setSuccess(true)
    window.setTimeout(() => setSuccess(false), 5000)
    // intervals/timeouts
    if (statsInterval.current) window.clearInterval(statsInterval.current)
    if (ccvFetchInterval.current) window.clearInterval(ccvFetchInterval.current)
    if (initialLoadTimeout.current) window.clearTimeout(initialLoadTimeout.current)
    statsInterval.current = window.setInterval(flushStats, 1000)
    window.setTimeout(fetchCCV, 3000)
    ccvFetchInterval.current = window.setInterval(fetchCCV, 120000)
    initialLoadTimeout.current = window.setTimeout(() => {
      initialLoadComplete.current = true
    }, 5000)
    // notify userscript
    window.setTimeout(() => {
      chatFrameRef.current?.contentWindow?.postMessage(
        { type: "HOLOCHATSTATS_VIDEO_LOADED", videoId },
        "https://www.youtube.com"
      )
    }, 2000)
  }
  // cleanup on unmount
  useEffect(() => {
    return () => {
      if (statsInterval.current) window.clearInterval(statsInterval.current)
      if (ccvFetchInterval.current) window.clearInterval(ccvFetchInterval.current)
      if (initialLoadTimeout.current) window.clearTimeout(initialLoadTimeout.current)
      if (connTimeout.current) window.clearTimeout(connTimeout.current)
    }
  }, [])
  // redraw graphs on resize
  useEffect(() => {
    let to: number
    const onResize = () => {
      window.clearTimeout(to)
      to = window.setTimeout(() => {
        drawGraph(ccvCanvasRef.current, ccvHistory.current, "rgb(72, 166, 167)")
        drawGraph(msgCanvasRef.current, msgPerMinHistory.current.map((h) => h.value), "rgb(92, 144, 210)")
      }, 100)
    }
    window.addEventListener("resize", onResize)
    return () => window.removeEventListener("resize", onResize)
  }, [])
  const TrendIcon = stats.ccvTrend === "up" ? ArrowUp : stats.ccvTrend === "down" ? ArrowDown : Minus
  return (
    <div className="viewer-root">
      {conn.show && (
        <div className={`connection-indicator ${conn.connected ? "connected" : "disconnected"} ${conn.fading ? "fade-out" : ""}`}>
          {conn.connected ? t("Userscript Connected") : t("Userscript Disconnected")}
        </div>
      )}
      {stage === "input" && (
        <div className="viewer-input-section">
          <div className="viewer-input-container">
            <h2 className="mb-4 text-2xl font-bold">{t("Live Chat Viewer")}</h2>
            <p className="mb-4">
              {t("Enter a YouTube live stream URL to view the video with chat and statistics. VODs are not supported and the HyperChat browser extension must be disabled before using this feature.")}
            </p>
            <div className="viewer-input-group">
              <div className="flex gap-2">
                <Input
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && loadVideo()}
                  placeholder="https://www.youtube.com/watch?v=..."
                  className="text-base"
                />
                <Button onClick={loadVideo}>
                  <PlayCircle className="mr-2 h-4 w-4" /> {t("Load")}
                </Button>
              </div>
            </div>
            <div className="viewer-userscript-info">
              <Download className="inline h-4 w-4 mr-1" />
              <strong>{t("For live chat statistics:")}</strong>{" "}
              <a href="https://greasyfork.org/en/scripts/558153-holochatstats-live-chat-tracker" target="_blank" rel="noreferrer">
                {t("Install the Userscript")}
              </a>
              <small className="block mt-2">
                {t("Requires Greasemonkey (Firefox) or Tampermonkey (Chrome/Edge)")}
              </small>
            </div>
            {error && (
              <div className="mt-5 rounded-lg bg-destructive px-4 py-3 text-sm">{error}</div>
            )}
            {success && (
              <div className="mt-5 rounded-lg bg-green-600 px-4 py-3 text-sm flex items-center justify-center gap-2">
                <CheckCircle className="h-4 w-4" />
                {t("Video loaded! Chat statistics will appear if the userscript is installed.")}
              </div>
            )}
          </div>
        </div>
      )}
      {stage === "viewing" && (
        <div className="viewer-video-section">
          <div className="video-chat-container">
            <div className="left-section">
              <div className="video-wrapper">
                <iframe
                  src={videoSrc}
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
                  allowFullScreen
                />
              </div>
              <div className="stats-area">
                <div className="stats-container">
                  <div className="stat-box">
                    <div className="stat-label">{t("Messages")}</div>
                    <div className="stat-value">{stats.messageCount.toLocaleString()}</div>
                  </div>
                  <div className="stat-box stat-box-with-graph">
                    <canvas ref={msgCanvasRef} className="stat-graph" />
                    <div className="stat-content">
                      <div className="stat-label">{t("Msgs/Min")}</div>
                      <div className="stat-value">{stats.messagesPerMinute}</div>
                    </div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">{t("Users")}</div>
                    <div className="stat-value">{stats.uniqueUsers.toLocaleString()}</div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">{t("Members")}</div>
                    <div className="stat-value">
                      {stats.memberCount.toLocaleString()}
                      <div className="stat-percentage">({stats.memberPercentage}%)</div>
                    </div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">{t("JP Messages")}</div>
                    <div className="stat-value">
                      {stats.jpCount.toLocaleString()}
                      <div className="stat-percentage">({stats.jpPercentage}%)</div>
                    </div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">{t("Gifts / Purchases")}</div>
                    <div className="stat-value">
                      {stats.giftCount.toLocaleString()} / {stats.purchaseCount.toLocaleString()}
                    </div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">{t("Est. Revenue")}</div>
                    <div className="stat-value">{stats.revenue}</div>
                  </div>
                  <div className="stat-box stat-box-with-graph">
                    <canvas ref={ccvCanvasRef} className="stat-graph" />
                    <div className="stat-content">
                      <div className="stat-label">{t("CCV")}</div>
                      <div className="stat-value">
                        {stats.ccv}
                        {stats.ccvTrend && (
                          <span className={`ccv-trend ${stats.ccvTrend}`}>
                            <TrendIcon className="inline h-3 w-3" />
                          </span>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="stat-box">
                    <div className="stat-label">{t("Time")}</div>
                    <div className="stat-value">{stats.elapsedTime}</div>
                  </div>
                </div>
              </div>
            </div>
            <div className="chat-wrapper">
              <iframe ref={chatFrameRef} src={chatSrc} />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}