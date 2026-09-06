type ContextGetter = () => unknown
const registry: { get: ContextGetter | null } = { get: null }
export function registerEriContext(fn: ContextGetter | null) {
  registry.get = fn
}
export function getEriPageContext(): unknown {
  return registry.get ? registry.get() : null
}