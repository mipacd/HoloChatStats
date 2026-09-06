// frontend/src/i18n.ts
import i18n from "i18next"
import { initReactI18next } from "react-i18next"
import LanguageDetector from "i18next-browser-languagedetector"
import en from "./locales/en/common.json"
import ja from "./locales/ja/common.json"
import ko from "./locales/ko/common.json"
i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      ja: { translation: ja },
      ko: { translation: ko },
    },
    fallbackLng: "en",
    supportedLngs: ["en", "ja", "ko"],
    load: "languageOnly",       // <-- strips region code: "en-US" -> "en"
    keySeparator: false,
    nsSeparator: false,
    interpolation: { escapeValue: false },
  })
export default i18n