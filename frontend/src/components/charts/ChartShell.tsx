import { useRef } from "react"
import { useTranslation } from "react-i18next"
import { Download, Info, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { downloadChartPng } from "@/lib/chart-export"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
interface ChartShellProps {
  title: string
  infoText?: string
  controls: React.ReactNode
  loading?: boolean
  hasData: boolean
  emptyText?: string
  children: React.ReactNode
  chartMinWidth?: number
  onDownloadCSV?: () => void
  png?: {
    title: string
    filename: string
    onBeforeCapture?: () => void
    onAfterCapture?: () => void
  }
  extraHeader?: React.ReactNode
}
const nextFrame = () =>
  new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
export function ChartShell({
  title,
  infoText,
  controls,
  loading,
  hasData,
  emptyText,
  children,
  chartMinWidth,
  onDownloadCSV,
  png,
  extraHeader,
}: ChartShellProps) {
  const { t } = useTranslation()
  const chartRef = useRef<HTMLDivElement>(null)
  const viewportRef = useRef<HTMLDivElement>(null)   // fixed-height box
  const scrollRef = useRef<HTMLDivElement>(null)     // overflow wrapper
  const handlePng = async () => {
    if (!chartRef.current || !png) return
    const viewport = viewportRef.current
    const scroll = scrollRef.current
    const chart = chartRef.current
    // Let the page un-clip its own scroll container first.
    png.onBeforeCapture?.()
    // Save current inline styles so we can restore them afterwards.
    const saved = {
      viewportHeight: viewport?.style.height ?? "",
      viewportMinHeight: viewport?.style.minHeight ?? "",
      scrollOverflowX: scroll?.style.overflowX ?? "",
      scrollOverflowY: scroll?.style.overflowY ?? "",
      chartHeight: chart.style.height ?? "",
    }
    // Expand everything to full content height for the snapshot.
    if (viewport) {
      viewport.style.height = "auto"
      viewport.style.minHeight = "0px"
    }
    if (scroll) {
      scroll.style.overflowX = "visible"
      scroll.style.overflowY = "visible"
    }
    chart.style.height = "auto"
    // Wait for React/layout to settle before capturing.
    await nextFrame()
    await nextFrame()
    try {
      await downloadChartPng(chart, png.title, png.filename)
    } finally {
      // Restore original styles.
      if (viewport) {
        viewport.style.height = saved.viewportHeight
        viewport.style.minHeight = saved.viewportMinHeight
      }
      if (scroll) {
        scroll.style.overflowX = saved.scrollOverflowX
        scroll.style.overflowY = saved.scrollOverflowY
      }
      chart.style.height = saved.chartHeight
      png.onAfterCapture?.()
    }
  }
  return (
    <TooltipProvider>
      <div className="flex flex-col gap-4">
        <h2 className="text-2xl font-bold text-center flex items-center justify-center gap-2">
          {title}
          {infoText && (
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-4 w-4 text-muted-foreground cursor-help" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">{infoText}</TooltipContent>
            </Tooltip>
          )}
        </h2>
        <div className="mx-auto flex flex-wrap items-end justify-center gap-4">{controls}</div>
        {extraHeader && <div className="flex justify-center">{extraHeader}</div>}
        <div
          style={{
            width: "95vw",
            marginLeft: "calc(50% - 47.5vw)",
          }}
        >
          {(onDownloadCSV || png) && (
            <div className="flex justify-end mb-2">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button size="sm" variant="outline" disabled={!hasData} aria-label={t("Download")}>
                    <Download className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  {onDownloadCSV && (
                    <DropdownMenuItem onClick={onDownloadCSV}>{t("Download CSV")}</DropdownMenuItem>
                  )}
                  {png && <DropdownMenuItem onClick={handlePng}>{t("Download PNG")}</DropdownMenuItem>}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )}
          <div
            ref={viewportRef}
            className="relative"
            style={{ height: "calc(100vh - 320px)", minHeight: 400 }}
          >
            {loading && (
              <div className="absolute inset-0 flex items-center justify-center z-10">
                <Loader2 className="h-10 w-10 animate-spin text-primary" />
              </div>
            )}
            {!loading && !hasData && (
              <div className="absolute inset-0 flex items-center justify-center text-muted-foreground">
                {emptyText ?? t("No data available for the selected criteria.")}
              </div>
            )}
            <div ref={scrollRef} className="h-full w-full overflow-x-auto overflow-y-hidden">
              <div
                ref={chartRef}
                className="h-full"
                style={{ minWidth: "100%", width: chartMinWidth ? `${chartMinWidth}px` : "100%" }}
              >
                {hasData && children}
              </div>
            </div>
          </div>
        </div>
      </div>
    </TooltipProvider>
  )
}