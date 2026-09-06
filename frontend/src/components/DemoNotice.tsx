import { AlertTriangle } from "lucide-react"
import { useTranslation } from "react-i18next"
export function DemoNotice() {
  const { t } = useTranslation()
  return (
    <div className="flex items-center justify-center gap-2 rounded-md border border-yellow-600/40 bg-yellow-950/40 text-yellow-200 text-sm p-2 text-center">
      <AlertTriangle className="h-4 w-4 flex-shrink-0" />
      <span>
        {t("This is a demo feature and will not be updated further. Data only covers August–October 2025.")}
      </span>
    </div>
  )
}