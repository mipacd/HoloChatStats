import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Loader2 } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { MonthPicker } from "@/components/ui/month-picker"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
interface LeaderboardRow {
  user_name: string
  message_count: number
}
export default function ChatLeaderboards() {
  const { t, i18n } = useTranslation()
  const [channels, setChannels] = useState<string[]>([])
  const [channel, setChannel] = useState("")
  const [month, setMonth] = useState("")
  const [rows, setRows] = useState<LeaderboardRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    api.get("/get_channel_names").then((res) => setChannels(res.data || []))
  }, [])
  useEffect(() => {
    registerEriContext(() => ({
      page: "chat_leaderboards",
      endpoint: "/api/get_chat_leaderboard",
      parameters: {
        month: month || new Date().toISOString().slice(0, 7),
        channel_name: channel,
      },
      description: `Viewing chat leaderboard data for ${channel} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [channel, month])
  const fetchLeaderboard = async () => {
    if (!channel || !month) {
      setError(t("Please fill in all fields."))
      return
    }
    setError(null)
    setLoading(true)
    setRows(null)
    try {
      const res = await api.get("/get_chat_leaderboard", {
        params: { channel_name: channel, month },
      })
      if (!res.data || res.data.length === 0) {
        setError(t("No data available for the selected criteria."))
      } else {
        setRows(res.data)
      }
    } catch {
      setError(t("Error fetching data. Please try again later."))
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="flex flex-col gap-6 max-w-2xl mx-auto w-full">
      <h2 className="text-2xl font-bold text-center">{t("Chat Leaderboards")}</h2>
      <Card>
        <CardContent className="pt-6">
          {/* Single row: channel + month + button all on one line */}
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex-1 min-w-[180px] space-y-1.5">
              <Label>{t("Channel:")}</Label>
              <Select value={channel} onValueChange={setChannel}>
                <SelectTrigger>
                  <SelectValue placeholder={t("Select Channel")} />
                </SelectTrigger>
                <SelectContent>
                  {channels.map((c) => (
                    <SelectItem key={c} value={c}>{c}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1 min-w-[160px] space-y-1.5">
              <Label>{t("Month:")}</Label>
              <MonthPicker
                value={month}
                onChange={setMonth}
                locale={i18n.language}
                className="max-w-none"
              />
            </div>
            <Button onClick={fetchLeaderboard} disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t("Get Leaderboard")}
            </Button>
          </div>
          {error && <p className="text-center text-destructive text-sm mt-4">{error}</p>}
        </CardContent>
      </Card>
      {rows && (
        <Card>
          <CardContent className="pt-6">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-10">#</TableHead>
                  <TableHead>{t("Username")}</TableHead>
                  <TableHead>{t("Message Count")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r, i) => (
                  <TableRow key={i}>
                    <TableCell>{i + 1}</TableCell>
                    <TableCell>{r.user_name}</TableCell>
                    <TableCell>{r.message_count}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}
    </div>
  )
}