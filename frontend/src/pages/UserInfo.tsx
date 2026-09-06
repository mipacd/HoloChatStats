import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Info, Loader2 } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { MonthPicker } from "@/components/ui/month-picker"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
interface UserRow {
  channel_name: string
  message_count: number
  percentile: number
}
export default function UserInfo() {
  const { t, i18n } = useTranslation()
  const [identifier, setIdentifier] = useState("")
  const [month, setMonth] = useState("")
  const [rows, setRows] = useState<UserRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    registerEriContext(() => ({
      page: "user_info",
      endpoint: "/api/get_user_info",
      parameters: { identifier, month },
      description: `Viewing message frequencies for user: ${identifier} in ${month}`,
    }))
    return () => registerEriContext(null)
  }, [identifier, month])
  const fetchStats = async () => {
    if (!identifier.trim() || !month) {
      setError(t("Please enter a Channel ID or handle and select a month."))
      return
    }
    setError(null)
    setLoading(true)
    setRows(null)
    try {
      const res = await api.get("/get_user_info", {
        params: { identifier: identifier.trim(), month },
      })
      if (!res.data.success || !res.data.data || res.data.data.length === 0) {
        setError(t("No data available for the selected user and month."))
      } else {
        setRows(res.data.data)
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || t("Error fetching data. Please try again later."))
    } finally {
      setLoading(false)
    }
  }
  return (
    <div className="flex flex-col gap-6 max-w-3xl mx-auto w-full">
      <h2 className="text-2xl font-bold text-center">{t("Message Frequencies by User")}</h2>
      <Alert>
        <Info className="h-4 w-4" />
        <AlertDescription>
          Your YouTube Channel ID can be found at the end of the URL when you visit the{" "}
          <a href="https://studio.youtube.com/" target="_blank" rel="noreferrer" className="underline">
            YouTube Studio
          </a>{" "}
          page (e.g., UC...). Handles (e.g., @username) may not yet be in the database, so providing a
          Channel ID is the most reliable method.
        </AlertDescription>
      </Alert>
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-wrap items-end gap-4">
            <div className="flex-1 min-w-[200px] space-y-1.5">
              <Label>{t("Channel ID or @handle:")}</Label>
              <Input
                value={identifier}
                onChange={(e) => setIdentifier(e.target.value)}
                placeholder={t("Enter Channel ID or handle")}
                onKeyDown={(e) => e.key === "Enter" && fetchStats()}
              />
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
            <Button onClick={fetchStats} disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              {t("Get Data")}
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
                  <TableHead>{t("Channel")}</TableHead>
                  <TableHead>{t("Message Count")}</TableHead>
                  <TableHead>{t("Percentile")}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((r, i) => (
                  <TableRow key={i}>
                    <TableCell>{r.channel_name}</TableCell>
                    <TableCell>{r.message_count}</TableCell>
                    <TableCell>{r.percentile}%</TableCell>
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