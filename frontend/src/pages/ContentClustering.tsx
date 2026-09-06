import { useTranslation } from "react-i18next"
import { ClusteringPage } from "./ClusteringPage"
export default function ContentClustering() {
  const { t } = useTranslation()
  return (
    <ClusteringPage
      pageId="content_clustering"
      endpoint="/content_clustering"
      title={t("Content Similarity Graph")}
      infoText={t("Determined by stream title similarity. Click and drag to zoom. Double click to reset.")}
    />
  )
}