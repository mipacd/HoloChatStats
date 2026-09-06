import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { Info, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { MonthPicker } from "@/components/ui/month-picker"
import { SimilarityGraph } from "@/components/clustering/SimilarityGraph"
import type { GraphNode, GraphLink } from "@/components/clustering/SimilarityGraph"
import { api } from "@/lib/api"
import { registerEriContext } from "@/components/eri/eri-context"
interface ClusteringPageProps {
  pageId: string
  endpoint: string // e.g. "/channel_clustering"
  title: string
  infoText: string
}
export function ClusteringPage({ pageId, endpoint, title, infoText }: ClusteringPageProps) {
  const { t, i18n } = useTranslation()
  const [month, setMonth] = useState("")
  const [threshold, setThreshold] = useState("95")
  const [view3d, setView3d] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [graph, setGraph] = useState<{ nodes: GraphNode[]; links: GraphLink[] } | null>(null)
  useEffect(() => {
    registerEriContext(() => ({
      page: pageId,
      endpoint: `/api${endpoint}`,
      parameters: { month, percentile: threshold, type: view3d ? "3d" : "2d" },
      description: `Viewing ${pageId} graph for ${month}`,
    }))
    return () => registerEriContext(null)
  }, [pageId, endpoint, month, threshold, view3d])
  const loadGraph = async () => {
    if (!month) {
      setError(t("Please select a month."))
      return
    }
    setError(null)
    setLoading(true)
    try {
      const res = await api.get(endpoint, {
        params: { month, percentile: threshold, type: view3d ? "3d" : "2d" },
      })
      if (res.data.error) {
        setError(res.data.error)
        setGraph(null)
        return
      }
      setGraph(res.data)
    } catch {
      setError(t("Failed to load clustering graph."))
      setGraph(null)
    } finally {
      setLoading(false)
    }
  }
  return (
    <TooltipProvider>
      <div className="flex flex-col gap-4">
        <h2 className="text-2xl font-bold text-center flex items-center justify-center gap-2">
          {title}
          <Tooltip>
            <TooltipTrigger asChild>
              <Info className="h-4 w-4 text-muted-foreground cursor-help" />
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">{infoText}</TooltipContent>
          </Tooltip>
        </h2>
        <div className="flex flex-wrap items-end justify-center gap-4">
          <div className="space-y-1.5">
            <Label>{t("Select Month:")}</Label>
            <MonthPicker value={month} onChange={setMonth} locale={i18n.language} />
          </div>
          <div className="space-y-1.5">
            <Label>{t("Threshold:")}</Label>
            <Select value={threshold} onValueChange={setThreshold}>
              <SelectTrigger className="w-[220px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="95">{t("High (Top 5%)")}</SelectItem>
                <SelectItem value="90">{t("Medium (Top 10%)")}</SelectItem>
                <SelectItem value="80">{t("Low (Top 20%)")}</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div className="flex items-center gap-2 pb-2">
            <Switch id={`${pageId}-view3d`} checked={view3d} onCheckedChange={setView3d} />
            <Label htmlFor={`${pageId}-view3d`} className="flex items-center gap-1">
              {t("3D View:")}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Info className="h-3.5 w-3.5 text-muted-foreground cursor-help" />
                </TooltipTrigger>
                <TooltipContent>{t("Click and drag to rotate. Mouse wheel to zoom.")}</TooltipContent>
              </Tooltip>
            </Label>
          </div>
          <Button onClick={loadGraph} disabled={loading}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            {t("Load Clustering Graph")}
          </Button>
        </div>
        {error && <p className="text-center text-destructive text-sm">{error}</p>}
        <div
          className="relative"
          style={{
            width: "95vw",
            marginLeft: "calc(50% - 47.5vw)",
            height: "calc(100vh - 260px)",
            minHeight: 400,
          }}
        >
          {graph ? (
            <SimilarityGraph nodes={graph.nodes} links={graph.links} is3d={view3d} loading={loading} />
          ) : (
            loading && (
              <div className="absolute inset-0 flex items-center justify-center">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
              </div>
            )
          )}
        </div>
      </div>
    </TooltipProvider>
  )
}