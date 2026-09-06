import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Info, Loader2 } from "lucide-react"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Input } from "@/components/ui/input"
import { Alert, AlertDescription } from "@/components/ui/alert"
import {
  Tooltip, TooltipContent, TooltipProvider, TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
interface RecRow {
  channel_name: string
  score: number
}
export default function RecommendationEngine() {
  const { t } = useTranslation()
  const [identifier, setIdentifier] = useState("")
  const [rows, setRows] = useState<RecRow[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    registerEriContext(() => ({
      page: "recommendation_engine",
      endpoint: "/api/recommend",
      parameters: { identifier },
      description: `Viewing recommendations for identifier: ${identifier}`,
    }))
    return () => registerEriContext(null)
  }, [identifier])
  const fetchRecs = async () => {
    if (!identifier.trim()) {
      setError(t("Please enter a Channel ID or handle."))
      return
    }
    setError(null)
    setLoading(true)
    setRows(null)
    try {
      const res = await api.get("/recommend", {
        params: { identifier: identifier.trim(), months: 6 },
      })
      if (res.data.error) {
        setError(res.data.error)
      } else if (!res.data.recommended_channels || res.data.recommended_channels.length === 0) {
        setError(
          t("No recommendations could be generated. This usually means there is not enough recent chat history for the user.")
        )
      } else {
        setRows(res.data.recommended_channels)
      }
    } catch (err: any) {
      setError(err?.response?.data?.error || t("Error fetching data. Please try again later."))
    } finally {
      setLoading(false)
    }
  }
  return (
    <TooltipProvider>
      <div className="flex flex-col gap-6 max-w-2xl mx-auto w-full">
        <h2 className="text-2xl font-bold text-center flex items-center justify-center gap-2">
          {t("Recommendation Engine")}
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-4 w-4 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              {t("Calculated using chat data from the previous 6 months. Only shows channels where the user has chatted 3 times or less. Takes ~30 seconds to generate.")}
            </TooltipContent>
          </Tooltip>
        </h2>
        <Alert>
          <Info className="h-4 w-4" />
          <AlertDescription>
            Your YouTube Channel ID can be found at the end of the URL when you visit the{" "}
            <a href="https://studio.youtube.com/" target="_blank" rel="noreferrer" className="underline">
              YouTube Studio
            </a>{" "}
            customization page. Providing a Channel ID is the most reliable method.
          </AlertDescription>
        </Alert>
        <Card>
          <CardContent className="pt-6">
            <div className="flex flex-wrap items-end gap-4">
              <div className="flex-1 min-w-[200px] space-y-1.5">
                <Label>{t("Your Channel ID or @handle:")}</Label>
                <Input
                  value={identifier}
                  onChange={(e) => setIdentifier(e.target.value)}
                  placeholder={t("Enter Channel ID or handle")}
                  onKeyDown={(e) => e.key === "Enter" && fetchRecs()}
                />
              </div>
              <Button onClick={fetchRecs} disabled={loading}>
                {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
                {t("Get Recommendations")}
              </Button>
            </div>
            {loading && (
              <p className="text-center text-muted-foreground text-sm mt-4">
                {t("This can take ~30 seconds to generate...")}
              </p>
            )}
            {error && <p className="text-center text-destructive text-sm mt-4">{error}</p>}
          </CardContent>
        </Card>
        {rows && (
          <Card>
            <CardContent className="pt-6">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t("Recommended Channel")}</TableHead>
                    <TableHead>{t("Similarity Score (0-100)")}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {rows.map((r, i) => (
                    <TableRow key={i}>
                      <TableCell>{r.channel_name}</TableCell>
                      <TableCell>{r.score.toFixed(2)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}
      </div>
    </TooltipProvider>
  )
}