import { useTranslation } from "react-i18next"
import { StreamingHoursPage } from "./StreamingHoursPage"
export default function StreamingHoursMax() {
  const { t } = useTranslation()
  return (
    <StreamingHoursPage
      pageId="streaming_hours_max"
      endpoint="/get_group_max_streaming_hours"
      title={t("Longest Stream Duration")}
      infoText={t("Archived, non-member streams from YouTube only.")}
      pngTitlePrefix="Longest Stream Duration"
      csvFilename="streaming_hours_max.csv"
      pngFilename="max_streaming_hours.png"
    />
  )
}