import { useCallback, useEffect, useRef, useState } from "react"
import { useTranslation } from "react-i18next"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { Loader2, Mic, Send, Volume2 } from "lucide-react"
import {
  Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import { HelpSidebar } from "./HelpSidebar"
import { samplePromptGroups } from "./samplePrompts"
import "./eri.css"
const BASE_URL = (import.meta.env.VITE_ERI_API_URL || "/llm").replace(/\/$/, "");
type Sender = "eri" | "user"
interface Message {
  id: number
  sender: Sender
  text: string
  markdown?: boolean
}
let msgId = 0
export default function Eri() {
  const { t } = useTranslation()
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
  const [promptsRemaining, setPromptsRemaining] = useState<number | null>(null)
  const [dailyLimit, setDailyLimit] = useState(10)
  const [modelStatus, setModelStatus] = useState<"green" | "yellow" | "red">("red")
  const [overlayImg, setOverlayImg] = useState<string | null>(null)
  const [promptSelectValue, setPromptSelectValue] = useState("")
  const messagesRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const historyRef = useRef<{ role: string; content: string }[]>([])
  const recognitionRef = useRef<any>(null)
  const recordingRef = useRef(false)
  const synthRef = useRef<SpeechSynthesis | null>(
    typeof window !== "undefined" ? window.speechSynthesis : null
  )
  const lastEriRef = useRef(messages[0].text)
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
  const scrollToBottom = () =>
    requestAnimationFrame(() => {
      if (messagesRef.current) messagesRef.current.scrollTop = messagesRef.current.scrollHeight
    })
  const appendMessage = useCallback((sender: Sender, text: string, markdown = false) => {
    setMessages((prev) => [...prev, { id: msgId++, sender, text, markdown }])
    if (sender === "eri") lastEriRef.current = text.replace(/!\[[^\]]*\]\([^)]*\)/g, "").trim()
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
    const base = (navigator.language || "en-US").split("-")[0]
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
      finalTranscript = inputRef.current?.value || ""
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
  // ---- prompts remaining / model status ----
  const exhausted = promptsRemaining !== null && promptsRemaining <= 0 && !isAdmin
  const promptLevel = (() => {
    if (promptsRemaining === null) return ""
    if (promptsRemaining <= 0) return "level-exhausted"
    if (promptsRemaining <= Math.ceil(dailyLimit * 0.2)) return "level-danger"
    if (promptsRemaining <= Math.ceil(dailyLimit * 0.5)) return "level-warning"
    return "level-good"
  })()
  const fetchPromptsRemaining = useCallback(async () => {
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
  useEffect(() => {
    fetchPromptsRemaining()
    fetchModelStatus()
    const id = setInterval(fetchModelStatus, 60000)
    return () => clearInterval(id)
  }, [fetchPromptsRemaining, fetchModelStatus])
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
    historyRef.current.push({ role: "user", content: message })
    try {
      const payload: any = { message, chat_history: historyRef.current }
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
            if (!isAdmin) fetchPromptsRemaining()
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
      inputRef.current?.focus()
    }
  }
  // ---- markdown ----
  const mdComponents = {
    a: ({ node, ...props }: any) => <a {...props} target="_blank" rel="noopener noreferrer" />,
    img: ({ node, ...props }: any) => {
      let src: string = props.src || ""
      const chart = src.match(/(?:^|\/)charts\/([^?#]+\.png(?:[?#].*)?)$/i)
      if (chart) src = `${BASE_URL}/charts/${chart[1]}`
      if (src.includes("/charts/")) {
        return <img {...props} src={src} onClick={() => setOverlayImg(src)} />
      }
      return (
        <a href={src} target="_blank" rel="noopener noreferrer">
          <img {...props} src={src} />
        </a>
      )
    },
  }
  return (
    <div className="eri-page">
      <div className="eri-chat-container">
        <div className="eri-chat-messages" ref={messagesRef}>
          {messages.map((m) => (
            <div key={m.id} className={`eri-message-row ${m.sender}`}>
              <div className={`eri-bubble ${m.sender}`}>
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
          {busy && <div className="eri-loading">{statusText}</div>}
        </div>
        <div className="eri-chat-input">
          {isAdmin && (
            <span title="Admin mode active" className="flex items-center text-lg">🛡️</span>
          )}
          <div className="eri-input-wrapper">
            <input
              ref={inputRef}
              type="text"
              value={input}
              disabled={busy || exhausted}
              placeholder={exhausted ? t("Daily prompt limit reached. Please try again tomorrow.") : t("Ask Eri a question...")}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") sendMessage()
              }}
              aria-label="Message"
            />
            <div className="eri-inline-buttons">
              {speechSupported && (
                <button
                  className={`eri-inline-btn${isRecording ? " recording" : ""}`}
                  title={isRecording ? t("Stop recording") : t("Voice input")}
                  disabled={busy || exhausted}
                  onClick={toggleRecording}
                >
                  <Mic className="h-4 w-4" />
                </button>
              )}
              {synthRef.current && (
                <button
                  className={`eri-inline-btn${ttsEnabled ? " speaking" : ""}`}
                  title={ttsEnabled ? t("Stop reading") : t("Read aloud")}
                  onClick={toggleTTS}
                >
                  <Volume2 className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>
          <div className="eri-send-group">
            <button className="eri-send-btn" disabled={busy || exhausted} onClick={sendMessage} aria-label="Send">
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </button>
            {!isAdmin && promptsRemaining !== null && (
              <span className={`eri-prompts-remaining eri-send-top ${promptLevel}`} title={t("Prompts remaining today")}>
                💬 {promptsRemaining}
              </span>
            )}
            <span
              className={`eri-status-dot eri-send-bottom status-${modelStatus}`}
              title={statusTooltips[modelStatus]}
            />
          </div>
        </div>
        <div className="mt-2">
          <Select
            value={promptSelectValue}
            onValueChange={(v) => {
              setInput(v)
              setPromptSelectValue("")
              inputRef.current?.focus()
            }}
          >
            <SelectTrigger className="w-full bg-neutral-900 border-neutral-700">
              <SelectValue placeholder={t("💡 Try one of these prompts...")} />
            </SelectTrigger>
            <SelectContent>
              {samplePromptGroups.map((group) => (
                <SelectGroup key={group.label}>
                  <SelectLabel>{t(group.label)}</SelectLabel>
                  {group.prompts.map((p) => (
                    <SelectItem key={p} value={p}>
                      {t(p)}
                    </SelectItem>
                  ))}
                </SelectGroup>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="mt-3 text-center text-xs text-neutral-400">
          <p className="mb-1">
            {t("Eri may display inaccurate information, so double-check its responses. Your conversations are logged to help make Eri better. By using Eri, you agree to this.")}
          </p>
          <p className="mb-0">{t("Conversations in English may be more accurate than in other languages.")}</p>
        </div>
      </div>
      <HelpSidebar />
      {overlayImg && (
        <div className="eri-chart-overlay" onClick={() => setOverlayImg(null)}>
          <img src={overlayImg} alt="Chart" />
        </div>
      )}
    </div>
  )
}
