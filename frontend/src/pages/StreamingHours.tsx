import { useTranslation } from "react-i18next"
import { StreamingHoursPage } from "./StreamingHoursPage"
export default function StreamingHours() {
  const { t } = useTranslation()
  return (
    <StreamingHoursPage
      pageId="streaming_hours"
      endpoint="/get_group_total_streaming_hours"
      title={t("Total Streaming Hours")}
      infoText={t("Archived, non-member streams from YouTube only.")}
      pngTitlePrefix="Total Streaming Hours"
      csvFilename="streaming_hours.csv"
      pngFilename="total_streaming_hours.png"
      defaultGroup="Hololive"
    />
  )
}