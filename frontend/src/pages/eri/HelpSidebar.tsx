import { useState } from "react"
import { useTranslation } from "react-i18next"
import { HelpCircle } from "lucide-react"
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import {
  Accordion, AccordionContent, AccordionItem, AccordionTrigger,
} from "@/components/ui/accordion"
import { helpSections } from "./helpContent"
export function HelpSidebar() {
  const { t } = useTranslation()
  const [open, setOpen] = useState(false)
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        title={t("Help & Instructions")}
        className={`fixed bottom-8 right-8 z-[1000] flex h-12 w-12 items-center justify-center rounded-full bg-[#1e90ff] text-white shadow-lg transition-all hover:scale-110 hover:bg-[#0066cc] ${
          open ? "pointer-events-none scale-75 opacity-0" : "opacity-100"
        }`}
      >
        <HelpCircle className="h-6 w-6" />
      </button>
      <Sheet open={open} onOpenChange={setOpen}>
        <SheetContent
          side="right"
          className="w-full overflow-y-auto border-white/10 bg-[#141414]/95 backdrop-blur-md sm:max-w-[400px]"
        >
          <SheetHeader>
            <SheetTitle className="text-white">{t("Help & Instructions")}</SheetTitle>
          </SheetHeader>
          <div className="mt-4 flex flex-col gap-6 pb-8">
            {helpSections.map((section) => (
              <div key={section.header}>
                <h3 className="mb-3 border-b border-[#1e90ff]/30 pb-2 text-base font-semibold text-[#1e90ff]">
                  {t(section.header)}
                </h3>
                <Accordion type="single" collapsible className="flex flex-col gap-2">
                  {section.items.map((item, i) => (
                    <AccordionItem
                      key={item.title}
                      value={`${section.header}-${i}`}
                      className="rounded-lg border-none bg-white/5 px-3"
                    >
                      <AccordionTrigger className="text-sm hover:no-underline">
                        <span className="flex items-center gap-2">
                          {item.icon && <item.icon className="h-4 w-4" />}
                          {t(item.title)}
                        </span>
                      </AccordionTrigger>
                      <AccordionContent className="text-sm text-muted-foreground">
                        {Array.isArray(item.body)
                          ? item.body.map((p, j) => (
                              <p key={j} className={j > 0 ? "mt-2" : ""}>{t(p)}</p>
                            ))
                          : <p>{t(item.body)}</p>}
                      </AccordionContent>
                    </AccordionItem>
                  ))}
                </Accordion>
              </div>
            ))}
          </div>
        </SheetContent>
      </Sheet>
    </>
  )
}