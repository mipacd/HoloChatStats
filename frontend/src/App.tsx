import { useEffect } from "react"
import { BrowserRouter, Navigate, Routes, Route, Outlet, useLocation } from "react-router-dom"
import { Navbar } from "@/components/Navbar"
import { EriWidget } from "@/components/eri/EriWidget"
import Home from "@/pages/Home"
import CommonStats from "@/pages/CommonStats"
import Heatmap from "@/pages/Heatmap"
import ChannelClustering from "@/pages/ChannelClustering"
import ContentClustering from "@/pages/ContentClustering"
import ChatLeaderboards from "@/pages/ChatLeaderboards"
import UserInfo from "@/pages/UserInfo"
import RecommendationEngine from "@/pages/RecommendationEngine"
import UserChange from "@/pages/UserChange"
import ExclusiveChat from "@/pages/ExclusiveChat"
import Engagement from "@/pages/Engagement"
import MembershipCounts from "@/pages/MembershipCounts"
import MembershipPercentages from "@/pages/MembershipPercentages"
import MembershipChange from "@/pages/MembershipChange"
import ChatMakeup from "@/pages/ChatMakeup"
import MessageTypes from "@/pages/MessageTypes"
import JpUserPercents from "@/pages/JpUserPercents"
import StreamingHours from "@/pages/StreamingHours"
import StreamingHoursAvg from "@/pages/StreamingHoursAvg"
import StreamingHoursMax from "@/pages/StreamingHoursMax"
import StreamingHoursDiff from "@/pages/StreamingHoursDiff"
import MonthlyStreamingHours from "@/pages/MonthlyStreamingHours"
import Viewer from "@/pages/Viewer"
import FunniestTimestamps from "@/pages/FunniestTimestamps"
import Highlights from "@/pages/Highlights"
import HighlightSearch from "@/pages/HighlightSearch"
import Eri from "@/pages/eri/Eri"
import SiteMetrics from "@/pages/SiteMetrics"
import StreamFrequencyPage from "@/pages/StreamFrequencyPage"
import StreamStats from "@/pages/StreamStats"

function PaddedLayout() {
  return (
    <main className="container mx-auto max-w-7xl px-4 py-6">
      <Outlet />
    </main>
  )
}

function PageViewTracker() {
  const { pathname } = useLocation()
  useEffect(() => {
    void fetch("/api/metrics/page-view", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ path: pathname }),
      keepalive: true,
    }).catch(() => undefined)
  }, [pathname])
  return null
}

export default function App() {
  return (
    <BrowserRouter>
      <PageViewTracker />
      <Navbar />
        <Routes>
          <Route element={<PaddedLayout />}>
            <Route path="/" element={<Home />} />
            <Route path="/common_users_members" element={<CommonStats />} />
            <Route path="/common_users" element={<Navigate to="/common_users_members?mode=users" replace />} />
            <Route path="/common_members" element={<Navigate to="/common_users_members?mode=members" replace />} />
            <Route path="/heatmap" element={<Heatmap />} />
            <Route path="/channel_clustering" element={<ChannelClustering />} />
            <Route path="/content_similarity" element={<ContentClustering />} />
            <Route path="/chat_leaderboards" element={<ChatLeaderboards />} />
            <Route path="/user_info" element={<UserInfo />} />
            <Route path="/recommendation_engine" element={<RecommendationEngine />} />  
            <Route path="/user_change" element={<UserChange />} />
            <Route path="/exclusive_chat" element={<ExclusiveChat />} />
            <Route path="/engagement" element={<Engagement />} />
            <Route path="/membership_counts" element={<MembershipCounts />} />
            <Route path="/membership_percentages" element={<MembershipPercentages />} />
            <Route path="/membership_change" element={<MembershipChange />} />
            <Route path="/chat_makeup" element={<ChatMakeup />} />
            <Route path="/message_types" element={<MessageTypes />} />
            <Route path="/jp_user_percents" element={<JpUserPercents />} />
            <Route path="/streaming_hours" element={<StreamingHours />} />
            <Route path="/streaming_hours_avg" element={<StreamingHoursAvg />} />
            <Route path="/streaming_hours_max" element={<StreamingHoursMax />} />
            <Route path="/streaming_hours_diff" element={<StreamingHoursDiff />} />
            <Route path="/monthly_streaming_hours" element={<MonthlyStreamingHours />} />
            <Route path="/stream_freq" element={<StreamFrequencyPage />} />
            <Route path="/stream_stats" element={<StreamStats />} />
            <Route path="/stream_stats/:videoId" element={<StreamStats />} />
            <Route path="/funniest_timestamps" element={<FunniestTimestamps />} />
            <Route path="/highlights" element={<Highlights />} />
            <Route path="/highlight_search" element={<HighlightSearch />} />
            <Route path="/site_metrics" element={<SiteMetrics />} />
            
          </Route>
        <Route path="/viewer" element={<Viewer />} />
        <Route path="/eri" element={<Eri />} />
        
        </Routes>
      <EriWidget />
    </BrowserRouter>
  )
}
