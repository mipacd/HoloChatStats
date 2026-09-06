import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
  videoId: string | null
  startSeconds: number | null
  title?: string
}
export function VideoModal({ open, onOpenChange, videoId, startSeconds, title }: Props) {
  // src is only set while open, so closing the dialog tears down the
  // iframe and stops playback — same behavior as the old hidden.bs.modal handler
  const src =
    open && videoId
      ? `https://www.youtube.com/embed/${videoId}?start=${startSeconds ?? 0}&autoplay=1`
      : ""
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-5xl w-[95vw] p-0 bg-black border-border overflow-hidden">
        <DialogHeader className="p-4 pb-0">
          <DialogTitle>{title ?? "Video Highlight"}</DialogTitle>
        </DialogHeader>
        <div className="relative w-full pt-[56.25%]">
          {src && (
            <iframe
              src={src}
              title="YouTube video player"
              className="absolute inset-0 w-full h-full"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}