import { useEffect, useMemo, useState } from "react"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import { api } from "@/lib/api"
interface MonthPickerProps {
  value?: string // "YYYY-MM"
  onChange: (value: string) => void
  placeholder?: string
  className?: string
  locale?: string
  availableMonths?: string[]
}
function getMonthNames(locale?: string, format: "short" | "long" = "short") {
  const dtf = new Intl.DateTimeFormat(locale, { month: format })
  return Array.from({ length: 12 }, (_, i) => dtf.format(new Date(2000, i, 1)))
}
export function MonthPicker({
  value,
  onChange,
  placeholder = "Select month",
  className,
  locale,
  availableMonths,
}: MonthPickerProps) {
  const [open, setOpen] = useState(false)
  const [latestPublishedMonth, setLatestPublishedMonth] = useState<string | null>(null)
  const initialYear = value ? parseInt(value.split("-")[0], 10) : new Date().getFullYear()
  const [viewYear, setViewYear] = useState(initialYear)
  const normalizedAvailable = useMemo(() => availableMonths === undefined
    ? null
    : Array.from(new Set(availableMonths.filter(month => /^\d{4}-(0[1-9]|1[0-2])$/.test(month))))
        .sort().reverse(), [availableMonths])
  const available = useMemo(() => normalizedAvailable === null
    ? null : new Set(normalizedAvailable), [normalizedAvailable])
  const availableYears = useMemo(() => normalizedAvailable === null
    ? []
    : Array.from(new Set(normalizedAvailable.map((month) => Number(month.slice(0, 4)))))
        .sort((a, b) => a - b), [normalizedAvailable])
  const monthNames = getMonthNames(locale, "short")
  const selectedMonth = value ? parseInt(value.split("-")[1], 10) - 1 : null
  const selectedYear = value ? parseInt(value.split("-")[0], 10) : null
  const label =
    value && selectedMonth !== null
      ? new Intl.DateTimeFormat(locale, { year: "numeric", month: "long" }).format(
          new Date(selectedYear!, selectedMonth, 1)
        )
      : placeholder
  useEffect(() => {
    if (availableMonths !== undefined) {
      const target = value || normalizedAvailable?.[0]
      if (target) setViewYear(Number(target.slice(0, 4)))
      return
    }
    let active = true
    api.get("/get_date_ranges").then((response) => {
      const maximum = response.data?.[1]
      const publishedMonth = typeof maximum === "string" ? maximum.slice(0, 7) : null
      if (!active || !publishedMonth) return
      setLatestPublishedMonth(publishedMonth)
      if (value && value > publishedMonth) onChange(publishedMonth)
    }).catch(() => undefined)
    return () => { active = false }
  }, [availableMonths, normalizedAvailable, onChange, value])
  const hasEarlierYear = available === null || availableYears.some((year) => year < viewYear)
  const hasLaterYear = available === null || availableYears.some((year) => year > viewYear)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          className={cn("w-full max-w-[300px] justify-start font-normal", className)}
        >
          {label}
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64 p-3">
        <div className="flex items-center justify-between mb-2">
          <Button variant="ghost" size="icon" disabled={!hasEarlierYear}
            onClick={() => setViewYear((y) => available === null ? y - 1
              : Math.max(...availableYears.filter((year) => year < y)))}>
            <ChevronLeft className="h-4 w-4" />
          </Button>
          <span className="font-medium">{viewYear}</span>
          <Button variant="ghost" size="icon" disabled={!hasLaterYear}
            onClick={() => setViewYear((y) => available === null ? y + 1
              : Math.min(...availableYears.filter((year) => year > y)))}>
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>
        <div className="grid grid-cols-3 gap-2">
          {monthNames.map((m, idx) => {
            const isSelected = selectedYear === viewYear && selectedMonth === idx
            const candidate = `${viewYear}-${String(idx + 1).padStart(2, "0")}`
            const isUnpublished = Boolean(latestPublishedMonth && candidate > latestPublishedMonth)
            const isUnavailable = available !== null && !available.has(candidate)
            return (
              <Button
                key={m}
                size="sm"
                variant={isSelected ? "default" : "ghost"}
                disabled={isUnavailable || (available === null && isUnpublished)}
                onClick={() => {
                  onChange(candidate)
                  setOpen(false)
                }}
              >
                {m}
              </Button>
            )
          })}
        </div>
      </PopoverContent>
    </Popover>
  )
}
