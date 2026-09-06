import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Loader2, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { api } from "@/lib/api"
interface Candidate {
  user_id: string
  username: string
}
interface Props {
  /** The month/group the *loaded* graph was built with, not the picker value. */
  month: string
  channelGroup: string
  /** Return an error string to keep the dialog open, or null on success. */
  onFound: (userIndex: number, username: string) => string | null
}
export function UserSearchDialog({ month, channelGroup, onFound }: Props) {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  const [term, setTerm] = useState("")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [candidates, setCandidates] = useState<Candidate[] | null>(null)
  const reset = () => {
    setTerm("")
    setError(null)
    setCandidates(null)
    setLoading(false)
  }
  const lookup = async (q: string) => {
    if (!q.trim() || loading) return
    setLoading(true)
    setError(null)
    setCandidates(null)
    try {
      const params: Record<string, string> = { month, q: q.trim() }
      if (channelGroup && channelGroup !== "all") params.channel_group = channelGroup
      const res = await api.get("/community_graph/find_user", { params })
      if (res.data.candidates?.length) {
        setCandidates(res.data.candidates)
        return
      }
      const err = onFound(res.data.index, res.data.username)
      if (err) {
        setError(err)
        return
      }
      setOpen(false)
      reset()
    } catch (e: any) {
      setError(e?.response?.data?.error ?? t("Lookup failed."))
    } finally {
      setLoading(false)
    }
  }
  return (
    <Dialog
      open={open}
      onOpenChange={(o) => {
        setOpen(o)
        if (!o) reset()
      }}
    >
      <DialogTrigger asChild>
        <button
          className="flex items-center gap-1.5 rounded-md border border-white/15 bg-zinc-900/90 px-3 py-1.5 text-xs text-zinc-300 shadow-lg backdrop-blur hover:bg-white/10 hover:text-white"
          title={t("Find a user in the graph")}
        >
          <Search className="h-3.5 w-3.5" />
          {t("Find user")}
        </button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>{t("Find user")}</DialogTitle>
          <DialogDescription>
            {t("Enter a YouTube channel ID (UC…) or a username / handle.")}
          </DialogDescription>
        </DialogHeader>
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault()
            lookup(term)
          }}
        >
          <Input
            autoFocus
            value={term}
            onChange={(e) => setTerm(e.target.value)}
            placeholder="UCxxxxxxxxxxxxxxxxxxxxxx  •  @handle"
          />
          <Button type="submit" disabled={loading || !term.trim()}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : t("Search")}
          </Button>
        </form>
        {error && <p className="text-sm text-destructive">{error}</p>}
        {candidates && (
          <div className="max-h-60 overflow-y-auto rounded-md border border-border">
            <p className="border-b border-border px-3 py-1.5 text-xs text-muted-foreground">
              {t("Multiple matches — pick one:")}
            </p>
            {candidates.map((c) => (
              <button
                key={c.user_id}
                disabled={loading}
                onClick={() => lookup(c.user_id)}
                className="flex w-full items-center justify-between gap-3 px-3 py-2 text-left text-sm hover:bg-accent"
              >
                <span className="truncate">{c.username}</span>
                <span className="shrink-0 font-mono text-[11px] text-muted-foreground">
                  {c.user_id}
                </span>
              </button>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}