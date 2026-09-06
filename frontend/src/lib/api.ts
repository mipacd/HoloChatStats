// frontend/src/lib/api.ts
import axios from "axios"
import i18n from "@/i18n"
export const api = axios.create({ baseURL: "/api" })
// attach current language so endpoints that return localized text
// (e.g. latest updates) can respond appropriately
api.interceptors.request.use((config) => {
  config.params = { ...config.params, lang: i18n.language }
  return config
})