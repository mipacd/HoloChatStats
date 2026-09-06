import { useState } from "react"
import { useTranslation } from "react-i18next"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ClusteringPage } from "./ClusteringPage"
import { CommunityGraphTab } from "@/components/clustering/CommunityGraphTab"
export default function ChannelClustering() {
  const { t } = useTranslation()
  const [activeTab, setActiveTab] = useState("similarity")
  return (
    <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
      <TabsList className="mx-auto flex w-fit">
        <TabsTrigger value="similarity">{t("Channel Similarity")}</TabsTrigger>
        <TabsTrigger value="community">{t("Community Graph")}</TabsTrigger>
      </TabsList>
      <TabsContent value="similarity" className="mt-2">
        <ClusteringPage
          pageId="channel_clustering"
          endpoint="/channel_clustering"
          title={t("Channel Clustering")}
          infoText={t(
            "Determined by common chat users using cosine similarity and Leiden community detection. Channels must meet a similarity threshold with at least one other channel to appear. Click and drag to zoom. Double click to reset."
          )}
        />
      </TabsContent>
      <TabsContent value="community" className="mt-2">
        <CommunityGraphTab />
      </TabsContent>
    </Tabs>
  )
}