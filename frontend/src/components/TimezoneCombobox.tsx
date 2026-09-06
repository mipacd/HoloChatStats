import { useMemo, useState } from "react"
import { Check, ChevronsUpDown } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList,
} from "@/components/ui/command"
import {
  Popover, PopoverContent, PopoverTrigger,
} from "@/components/ui/popover"
function getAllTimezones(): string[] {
  try {
    return (Intl as any).supportedValuesOf("timeZone") as string[]
  } catch {
    return [
      "UTC", "Pacific/Honolulu", "America/Anchorage", "America/Los_Angeles",
      "America/Denver", "America/Chicago", "America/New_York", "America/Sao_Paulo",
      "Europe/London", "Europe/Berlin", "Europe/Paris", "Europe/Moscow",
      "Asia/Dubai", "Asia/Kolkata", "Asia/Bangkok", "Asia/Jakarta",
      "Asia/Singapore", "Asia/Shanghai", "Asia/Hong_Kong", "Asia/Seoul",
      "Asia/Tokyo", "Australia/Perth", "Australia/Sydney", "Pacific/Auckland",
    ]
  }
}
function getUtcOffsetHours(tz: string): number {
  try {
    const now = new Date()
    const local = new Date(now.toLocaleString("en-US", { timeZone: tz }))
    const utc = new Date(now.toLocaleString("en-US", { timeZone: "UTC" }))
    return (local.getTime() - utc.getTime()) / 3_600_000
  } catch {
    return 0
  }
}
function formatOffset(hours: number): string {
  const sign = hours >= 0 ? "+" : "−"
  const abs = Math.abs(hours)
  const h = Math.floor(abs)
  const m = Math.round((abs - h) * 60)
  return `UTC${sign}${h}${m ? `:${String(m).padStart(2, "0")}` : ""}`
}
function formatTzName(tz: string): string {
  return tz.replace(/_/g, " ").replace(/\//g, " / ")
}
interface Props {
  value: string
  onChange: (tz: string) => void
  className?: string
}
export function TimezoneCombobox({ value, onChange, className }: Props) {
  const [open, setOpen] = useState(false)
  const zones = useMemo(() => {
    const all = getAllTimezones()
    if (!all.includes("UTC")) all.unshift("UTC")
    return all
      .map((tz) => ({
        tz,
        offset: getUtcOffsetHours(tz),
        label: `${formatTzName(tz)} (${formatOffset(getUtcOffsetHours(tz))})`,
      }))
      .sort((a, b) => a.offset - b.offset || a.tz.localeCompare(b.tz))
  }, [])
  const current = zones.find((z) => z.tz === value)
  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="outline"
          role="combobox"
          aria-expanded={open}
          className={cn("justify-between font-normal", className)}
        >
          <span className="truncate">
            {current ? current.label : formatTzName(value)}
          </span>
          <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-[320px] p-0" align="start">
        <Command>
          <CommandInput placeholder="Search timezone…" />
          <CommandList>
            <CommandEmpty>No timezone found.</CommandEmpty>
            <CommandGroup>
              {zones.map((z) => (
                <CommandItem
                  key={z.tz}
                  value={`${z.tz} ${z.label}`}
                  onSelect={() => {
                    onChange(z.tz)
                    setOpen(false)
                  }}
                >
                  <Check
                    className={cn(
                      "mr-2 h-4 w-4",
                      value === z.tz ? "opacity-100" : "opacity-0",
                    )}
                  />
                  {z.label}
                </CommandItem>
              ))}
            </CommandGroup>
          </CommandList>
        </Command>
      </PopoverContent>
    </Popover>
  )
}