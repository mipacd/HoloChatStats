import React from "react"
import ReactDOM from "react-dom/client"
import App from "./App"
import "./i18n"
import "./index.css"
import "flag-icons/css/flag-icons.min.css"
document.documentElement.classList.add("dark")
ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)