function esc(v: string | number) {
  const s = String(v)
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s
}
export function downloadCSV(
  filename: string,
  headers: (string | number)[],
  rows: (string | number)[][]
) {
  const lines = [headers.map(esc).join(","), ...rows.map((r) => r.map(esc).join(","))]
  const blob = new Blob([lines.join("\n")], { type: "text/csv;charset=utf-8;" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  link.click()
  URL.revokeObjectURL(url)
}
export async function downloadChartPng(
  node: HTMLElement,
  title: string,
  filename: string,
  bg = "#1a1a1a"
) {
  const { toPng } = await import("html-to-image")
  const scale = 2
  const dataUrl = await toPng(node, { backgroundColor: bg, pixelRatio: scale })
  const img = new Image()
  img.src = dataUrl
  await img.decode()
  const titleHeight = 60 * scale
  const canvas = document.createElement("canvas")
  canvas.width = img.width
  canvas.height = img.height + titleHeight
  const ctx = canvas.getContext("2d")!
  ctx.fillStyle = bg
  ctx.fillRect(0, 0, canvas.width, canvas.height)
  ctx.fillStyle = "white"
  ctx.font = `bold ${24 * scale}px Arial`
  ctx.textAlign = "center"
  ctx.fillText(title, canvas.width / 2, titleHeight / 2 + 10 * scale)
  ctx.drawImage(img, 0, titleHeight)
  const link = document.createElement("a")
  link.download = filename
  link.href = canvas.toDataURL("image/png")
  link.click()
}