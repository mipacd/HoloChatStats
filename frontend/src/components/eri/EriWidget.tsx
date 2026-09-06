import { useEffect, useRef, useState, useCallback } from "react"
import { useTranslation } from "react-i18next"
import { useLocation } from "react-router-dom"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import {
  BsChevronRight,
  BsChevronLeft,
  BsDashLg,
  BsVolumeUpFill,
  BsMicFill,
  BsSendFill,
  BsChatDots,
} from "react-icons/bs"
import { getEriPageContext } from "./eri-context"
import "./eri-widget.css"
type Sender = "eri" | "user"
interface Message {
  id: number
  sender: Sender
  text: string
  markdown?: boolean
}
const BASE_URL = (import.meta.env.VITE_ERI_API_URL || "/llm").replace(/\/$/, "");
let msgId = 0
export function EriWidget() {
  const { pathname } = useLocation()
  if (pathname === "/eri" || pathname === "/viewer") return null
  const { t } = useTranslation()
  // ---- UI state ----
  const [open, setOpen] = useState(false)
  const [minimizedToSide, setMinimizedToSide] = useState(false)
  const [bubbleVisible, setBubbleVisible] = useState(true)
  const [messages, setMessages] = useState<Message[]>([
    {
      id: msgId++,
      sender: "eri",
      text: t("Hey, I'm Eri - HoloChatStats' virtual assistant. How can I help you today?"),
    },
  ])
  const [input, setInput] = useState("")
  const [busy, setBusy] = useState(false)
  const [statusText, setStatusText] = useState("")
  const [isRecording, setIsRecording] = useState(false)
  const [ttsEnabled, setTtsEnabled] = useState(false)
  const [modelStatus, setModelStatus] = useState<"green" | "yellow" | "red">("red")
  const [promptsRemaining, setPromptsRemaining] = useState<number | null>(null)
  const [dailyLimit, setDailyLimit] = useState(10)
  const [overlayImg, setOverlayImg] = useState<string | null>(null)
  // ---- refs ----
  const windowRef = useRef<HTMLDivElement>(null)
  const messagesRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const historyRef = useRef<{ role: string; content: string }[]>([])
  const recognitionRef = useRef<any>(null)
  const synthRef = useRef<SpeechSynthesis | null>(
    typeof window !== "undefined" ? window.speechSynthesis : null
  )
  const lastEriRef = useRef<string>(messages[0].text)
  const hasBeenDragged = useRef(false)
  const urlParams = new URLSearchParams(window.location.search)
  const adminKey = urlParams.get("admin_key")
  const isAdmin = !!adminKey
  const statusMessages: Record<string, string> = {
    status_connecting: t("Let's see..."),
    status_translating: t("Deciphering your request..."),
    status_planner: t("Alright, plotting a course of action."),
    status_tools: t("Searching the HoloChatStats archives..."),
    status_responder: t("Compiling the data..."),
    rate_limit_exceeded: t("Looks like you've reached the daily query limit. Please try again tomorrow."),
  }
  const statusTooltips: Record<string, string> = {
    green: t("Model online"),
    yellow: t("Model degraded"),
    red: t("Model unavailable"),
  }
  // ---- helpers ----
  const scrollToBottom = () => {
    requestAnimationFrame(() => {
      if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight
    })
  }
  const appendMessage = useCallback((sender: Sender, text: string, markdown = false) => {
    setMessages((prev) => [...prev, { id: msgId++, sender, text, markdown }])
    if (sender === "eri") {
      lastEriRef.current = text.replace(/!\[[^\]]*\]\([^)]*\)/g, "").trim()
    }
    scrollToBottom()
  }, [])
  // ---- TTS ----
  const stripMarkdown = (s: string) =>
    s
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/\*([^*]+)\*/g, "$1")
      .replace(/__([^_]+)__/g, "$1")
      .replace(/_([^_]+)_/g, "$1")
      .replace(/`{3}[\s\S]*?`{3}/g, "")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/^\s*[-*+]\s+/gm, "")
      .replace(/^\s*\d+\.\s+/gm, "")
      .replace(/^\s*>\s+/gm, "")
      .replace(/!\[([^\]]*)\]\([^)]+\)/g, "")
      .replace(/\n{2,}/g, "\n")
      .trim()
  const getVoice = () => {
    const synth = synthRef.current
    if (!synth) return null
    const voices = synth.getVoices()
    const lang = navigator.language || "en-US"
    const base = lang.split("-")[0]
    const fem = ["female", "woman", "girl", "zira", "samantha", "victoria", "karen", "moira", "tessa", "fiona"]
    return (
      voices.find((v) => v.lang.startsWith(base) && fem.some((k) => v.name.toLowerCase().includes(k))) ||
      voices.find((v) => v.lang.startsWith(base)) ||
      voices.find((v) => v.lang.startsWith("en") && fem.some((k) => v.name.toLowerCase().includes(k))) ||
      voices.find((v) => v.lang.startsWith("en")) ||
      voices[0] ||
      null
    )
  }
  const speak = useCallback((text: string) => {
    const synth = synthRef.current
    if (!synth) return
    synth.cancel()
    const clean = stripMarkdown(text)
    if (!clean) return
    const u = new SpeechSynthesisUtterance(clean)
    const v = getVoice()
    if (v) {
      u.voice = v
      u.lang = v.lang
    }
    u.rate = 1
    u.pitch = 1.1
    u.volume = 1
    synth.speak(u)
  }, [])
  const stopSpeaking = useCallback(() => synthRef.current?.cancel(), [])
  const toggleTTS = () => {
    if (ttsEnabled) {
      setTtsEnabled(false)
      stopSpeaking()
    } else {
      setTtsEnabled(true)
      speak(lastEriRef.current)
    }
  }
  // speak newly arrived eri messages while TTS on
  useEffect(() => {
    if (ttsEnabled) speak(lastEriRef.current)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages])
  // ---- Speech recognition ----
  const initRecognition = useCallback(() => {
    const SR = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition
    if (!SR) return null
    const rec = new SR()
    rec.continuous = true
    rec.interimResults = true
    rec.lang = navigator.language || "en-US"
    let finalTranscript = ""
    rec.onstart = () => {
      setIsRecording(true)
      finalTranscript = textareaRef.current?.value || ""
    }
    rec.onresult = (e: any) => {
      let interim = ""
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const tr = e.results[i][0].transcript
        if (e.results[i].isFinal) finalTranscript += tr
        else interim += tr
      }
      setInput(finalTranscript + interim)
    }
    rec.onerror = (e: any) => {
      if (e.error === "not-allowed") {
        alert(t("Microphone access was denied. Please allow microphone access to use voice input."))
      }
      stopRecording()
    }
    rec.onend = () => {
      if (recordingRef.current) {
        try {
          rec.start()
        } catch {
          stopRecording()
        }
      }
    }
    return rec
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const recordingRef = useRef(false)
  useEffect(() => {
    recordingRef.current = isRecording
  }, [isRecording])
  const startRecording = () => {
    if (ttsEnabled) {
      setTtsEnabled(false)
      stopSpeaking()
    }
    if (!recognitionRef.current) recognitionRef.current = initRecognition()
    try {
      recognitionRef.current?.start()
    } catch {
      /* already started */
    }
  }
  const stopRecording = () => {
    setIsRecording(false)
    try {
      recognitionRef.current?.stop()
    } catch {
      /* noop */
    }
  }
  const toggleRecording = () => (isRecording ? stopRecording() : startRecording())
  const speechSupported =
    typeof window !== "undefined" &&
    ((window as any).SpeechRecognition || (window as any).webkitSpeechRecognition)
  // ---- autosize textarea ----
  useEffect(() => {
    const ta = textareaRef.current
    if (!ta) return
    ta.style.height = "auto"
    const h = Math.min(ta.scrollHeight, 120)
    ta.style.height = h + "px"
    ta.style.overflowY = ta.scrollHeight > 120 ? "auto" : "hidden"
  }, [input])
  // ---- drag & resize ----
  useEffect(() => {
    const win = windowRef.current
    if (!win) return
    const state = {
      dragging: false,
      resizing: false,
      handle: "",
      startX: 0,
      startY: 0,
      startLeft: 0,
      startTop: 0,
      startW: 0,
      startH: 0,
    }
    const pt = (e: MouseEvent | TouchEvent) =>
      "touches" in e ? e.touches[0] : (e as MouseEvent)
    const onMove = (e: MouseEvent | TouchEvent) => {
      const p = pt(e)
      if (!p) return
      if (state.dragging) {
        let left = state.startLeft + (p.clientX - state.startX)
        let top = state.startTop + (p.clientY - state.startY)
        const m = 20
        left = Math.max(m, Math.min(window.innerWidth - win.offsetWidth - m, left))
        top = Math.max(m, Math.min(window.innerHeight - win.offsetHeight - m, top))
        win.style.left = left + "px"
        win.style.top = top + "px"
      } else if (state.resizing) {
        const dx = p.clientX - state.startX
        const dy = p.clientY - state.startY
        let w = state.startW
        let h = state.startH
        let left = state.startLeft
        let top = state.startTop
        if (state.handle.includes("e")) w = state.startW + dx
        if (state.handle.includes("w")) {
          w = state.startW - dx
          left = state.startLeft + dx
        }
        if (state.handle.includes("s")) h = state.startH + dy
        if (state.handle.includes("n")) {
          h = state.startH - dy
          top = state.startTop + dy
        }
        if (w >= 300) {
          win.style.width = w + "px"
          if (state.handle.includes("w")) win.style.left = left + "px"
        }
        if (h >= 400) {
          win.style.height = h + "px"
          if (state.handle.includes("n")) win.style.top = top + "px"
        }
      }
    }
    const onUp = () => {
      state.dragging = false
      state.resizing = false
      win.classList.remove("dragging", "resizing")
    }
    const header = win.querySelector(".eri-chat-header") as HTMLElement
    const onHeaderDown = (e: MouseEvent | TouchEvent) => {
      const tgt = e.target as HTMLElement
      if (tgt.closest(".eri-minimize-btn") || tgt.closest(".eri-tts-btn") || tgt.closest(".eri-prompts-remaining"))
        return
      const p = pt(e)
      if (!p) return
      state.dragging = true
      hasBeenDragged.current = true
      win.classList.add("dragging")
      const r = win.getBoundingClientRect()
      state.startX = p.clientX
      state.startY = p.clientY
      state.startLeft = r.left
      state.startTop = r.top
      win.style.bottom = "auto"
      win.style.right = "auto"
      win.style.left = r.left + "px"
      win.style.top = r.top + "px"
      e.preventDefault()
    }
    const handleDowns: Array<[HTMLElement, (e: any) => void]> = []
    win.querySelectorAll<HTMLElement>(".eri-resize-handle").forEach((h) => {
      const dir = h.className.split(" ")[1].replace("eri-resize-", "")
      const fn = (e: MouseEvent | TouchEvent) => {
        const p = pt(e)
        if (!p) return
        state.resizing = true
        state.handle = dir
        win.classList.add("resizing")
        const r = win.getBoundingClientRect()
        state.startX = p.clientX
        state.startY = p.clientY
        state.startW = r.width
        state.startH = r.height
        state.startLeft = r.left
        state.startTop = r.top
        e.preventDefault()
        e.stopPropagation()
      }
      h.addEventListener("mousedown", fn)
      h.addEventListener("touchstart", fn)
      handleDowns.push([h, fn])
    })
    header?.addEventListener("mousedown", onHeaderDown)
    header?.addEventListener("touchstart", onHeaderDown)
    document.addEventListener("mousemove", onMove)
    document.addEventListener("touchmove", onMove)
    document.addEventListener("mouseup", onUp)
    document.addEventListener("touchend", onUp)
    return () => {
      header?.removeEventListener("mousedown", onHeaderDown)
      header?.removeEventListener("touchstart", onHeaderDown)
      document.removeEventListener("mousemove", onMove)
      document.removeEventListener("touchmove", onMove)
      document.removeEventListener("mouseup", onUp)
      document.removeEventListener("touchend", onUp)
      handleDowns.forEach(([h, fn]) => {
        h.removeEventListener("mousedown", fn)
        h.removeEventListener("touchstart", fn)
      })
    }
  }, [open])
  // ---- model status + prompts ----
  const fetchModelStatus = useCallback(async () => {
    try {
      const res = await fetch(`${BASE_URL}/model-status`)
      if (!res.ok) throw new Error()
      const data = await res.json()
      setModelStatus(data.status || "red")
    } catch {
      setModelStatus("red")
    }
  }, [])
  const fetchPrompts = useCallback(async () => {
    if (isAdmin) return
    try {
      const res = await fetch(`${BASE_URL}/prompts-remaining`)
      if (res.ok) {
        const data = await res.json()
        setDailyLimit(data.daily_limit)
        setPromptsRemaining(data.prompts_remaining)
      }
    } catch {
      setPromptsRemaining(null)
    }
  }, [isAdmin])
  useEffect(() => {
    fetchPrompts()
    fetchModelStatus()
    const id = setInterval(fetchModelStatus, 60000)
    const bubbleTimer = setTimeout(() => setBubbleVisible(false), 8000)
    return () => {
      clearInterval(id)
      clearTimeout(bubbleTimer)
    }
  }, [fetchModelStatus, fetchPrompts])
  const exhausted = promptsRemaining !== null && promptsRemaining <= 0 && !isAdmin
  const promptLevel = (() => {
    if (promptsRemaining === null) return ""
    if (promptsRemaining <= 0) return "level-exhausted"
    if (promptsRemaining <= Math.ceil(dailyLimit * 0.2)) return "level-danger"
    if (promptsRemaining <= Math.ceil(dailyLimit * 0.5)) return "level-warning"
    return "level-good"
  })()
  // ---- send ----
  const sendMessage = async () => {
    const message = input.trim()
    if (!message || busy || exhausted) return
    stopRecording()
    stopSpeaking()
    appendMessage("user", message)
    setInput("")
    setBusy(true)
    setStatusText(statusMessages.status_connecting)
    const pageContext = getEriPageContext()
    const enriched = pageContext
      ? `[Page Context: ${JSON.stringify(pageContext)}]\n${message}`
      : message
    historyRef.current.push({ role: "user", content: enriched })
    try {
      const payload: any = {
        message: enriched,
        chat_history: historyRef.current,
        page_context: pageContext,
      }
      if (adminKey) payload.admin_key = adminKey
      const res = await fetch(`${BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        if (res.status === 429) {
          appendMessage("eri", statusMessages.rate_limit_exceeded)
          if (!isAdmin) setPromptsRemaining(0)
        } else {
          appendMessage("eri", t("Sorry — I couldn't reach the assistant (network error)."))
        }
        setBusy(false)
        return
      }
      const reader = res.body!.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
      const handleEvent = (line: string) => {
        if (!line.startsWith("data:")) return
        try {
          const update = JSON.parse(line.slice(5).trim())
          if (update.type === "status") {
            setStatusText(statusMessages[update.key] || "Working...")
          } else if (update.type === "answer") {
            appendMessage("eri", update.message, true)
            historyRef.current.push({ role: "assistant", content: update.message })
            if (!isAdmin) fetchPrompts()
          } else if (update.type === "error") {
            setStatusText(update.message)
          }
        } catch {
          /* ignore malformed */
        }
      }
      for (;;) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const events = buffer.split("\n\n")
        buffer = events.pop() || ""
        events.forEach((ev) => handleEvent(ev.trim()))
      }
      if (buffer.trim()) handleEvent(buffer.trim())
    } catch {
      appendMessage("eri", t("Sorry — I couldn't reach the assistant (network error)."))
    } finally {
      setBusy(false)
      textareaRef.current?.focus()
    }
  }
  // ---- markdown renderers ----
  const mdComponents = {
    a: ({ node, ...props }: any) => <a {...props} target="_blank" rel="noopener noreferrer" />,
    img: ({ node, ...props }: any) => {
      let src: string = props.src || ""
      if (src.startsWith("/charts/")) src = BASE_URL + src
      if (src.includes("/charts/")) {
        return (
          <img
            {...props}
            src={src}
            data-eri-chart="true"
            onClick={() => setOverlayImg(src)}
            style={{ cursor: "zoom-in" }}
          />
        )
      }
      return (
        <a href={src} target="_blank" rel="noopener noreferrer">
          <img {...props} src={src} />
        </a>
      )
    },
  }
  // ---- render ----
  return (
    <>
      <div
        className={`eri-chat-widget${minimizedToSide ? " minimized-to-side" : ""}`}
        id="eriChatWidget"
      >
        <div className={`eri-widget-trigger${open ? " hidden" : ""}`}>
          <button
            className={`eri-side-arrow${minimizedToSide ? " show" : ""}`}
            title={minimizedToSide ? t("Restore") : t("Minimize to side")}
            onClick={(e) => {
              e.stopPropagation()
              setMinimizedToSide((v) => !v)
            }}
          >
            {minimizedToSide ? <BsChevronLeft /> : <BsChevronRight />}
          </button>
          {bubbleVisible && <div className="eri-bubble-widget">{t("💬 Got a question?")}</div>}
          <img
            src="/eri_chibi.png"
            alt="Eri"
            className="eri-chibi-widget"
            onClick={() => {
              setOpen(true)
              setTimeout(() => textareaRef.current?.focus(), 0)
            }}
          />
        </div>
        <div ref={windowRef} className={`eri-chat-window${open ? " active" : ""}`}>
          <div className="eri-resize-handle eri-resize-n" />
          <div className="eri-resize-handle eri-resize-e" />
          <div className="eri-resize-handle eri-resize-s" />
          <div className="eri-resize-handle eri-resize-w" />
          <div className="eri-resize-handle eri-resize-ne" />
          <div className="eri-resize-handle eri-resize-se" />
          <div className="eri-resize-handle eri-resize-sw" />
          <div className="eri-resize-handle eri-resize-nw" />
          <div className="eri-chat-header">
            <div className="eri-chat-header-title">
              <img src="/eri_chibi.png" alt="Eri" />
              <span>{t("Chat with Eri")}</span>
              {isAdmin && (
                <span className="eri-admin-badge" title="Admin mode active">
                  🛡️
                </span>
              )}
            </div>
            <div className="eri-widget-status-group">
              <span
                className={`eri-widget-status-dot status-${modelStatus}`}
                title={statusTooltips[modelStatus]}
              />
              {!isAdmin && promptsRemaining !== null && (
                <span
                  className={`eri-prompts-remaining ${promptLevel}`}
                  title={t("Prompts remaining today")}
                  style={{ display: "flex" }}
                >
                  <BsChatDots />
                  <span>{promptsRemaining}</span>
                </span>
              )}
            </div>
            <div className="eri-header-buttons">
              {synthRef.current && (
                <button
                  className={`eri-tts-btn${ttsEnabled ? " speaking" : ""}`}
                  title={ttsEnabled ? t("Stop reading") : t("Read aloud")}
                  onClick={toggleTTS}
                >
                  <BsVolumeUpFill />
                </button>
              )}
              <button
                className="eri-minimize-btn"
                title={t("Minimize")}
                onClick={() => {
                  setOpen(false)
                  stopRecording()
                  setTtsEnabled(false)
                  stopSpeaking()
                }}
              >
                <BsDashLg />
              </button>
            </div>
          </div>
          <div className="eri-widget-messages" ref={messagesRef}>
            {messages.map((m) => (
              <div key={m.id} className={`eri-widget-message-row ${m.sender}`}>
                <div className={`eri-widget-bubble ${m.sender}`}>
                  {m.sender === "eri" && <span className="sender">{t("Eri")}</span>}
                  <div className="msg">
                    {m.markdown ? (
                      <ReactMarkdown remarkPlugins={[remarkGfm]} components={mdComponents}>
                        {m.text}
                      </ReactMarkdown>
                    ) : (
                      m.text
                    )}
                  </div>
                </div>
              </div>
            ))}
            {busy && (
              <div className="eri-widget-loading">
                <div className="eri-spinner" />
                <span>{statusText}</span>
              </div>
            )}
          </div>
          <div className="eri-widget-input-area">
            <div className="eri-widget-input-wrapper">
              <div className="eri-widget-textarea-container">
                <textarea
                  ref={textareaRef}
                  className="eri-widget-input"
                  placeholder={
                    exhausted
                      ? t("Daily prompt limit reached. Please try again tomorrow.")
                      : t("Ask Eri a question...")
                  }
                  autoComplete="off"
                  rows={1}
                  value={input}
                  disabled={busy || exhausted}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault()
                      sendMessage()
                    }
                  }}
                />
                <div className="eri-widget-inline-buttons">
                  {speechSupported && (
                    <button
                      className={`eri-widget-inline-btn${isRecording ? " recording" : ""}`}
                      title={isRecording ? t("Stop recording") : t("Voice input")}
                      disabled={busy || exhausted}
                      onClick={toggleRecording}
                    >
                      <BsMicFill />
                    </button>
                  )}
                </div>
              </div>
              <button
                className="eri-widget-send-btn"
                title={t("Send")}
                disabled={busy || exhausted}
                onClick={sendMessage}
              >
                <BsSendFill />
              </button>
            </div>
          </div>
        </div>
      </div>
      {overlayImg && (
        <div className="eri-chart-overlay" onClick={() => setOverlayImg(null)}>
          <img src={overlayImg} alt="Chart" />
        </div>
      )}
    </>
  )
}