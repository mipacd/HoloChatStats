import { useTranslation } from "react-i18next"
import { StreamingHoursPage } from "./StreamingHoursPage"
export default function StreamingHoursAvg() {
  const { t } = useTranslation()
  return (
    <StreamingHoursPage
      pageId="streaming_hours_avg"
      endpoint="/get_group_avg_streaming_hours"
      title={t("Average Streaming Hours")}
      infoText={t("Archived, non-member streams from YouTube only.")}
      pngTitlePrefix="Avg. Streaming Hours"
      csvFilename="streaming_hours_avg.csv"
      pngFilename="avg_streaming_hours.png"
    />
  )
}