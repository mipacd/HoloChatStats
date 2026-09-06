import ForceGraph3D from "react-force-graph-3d"
import SpriteText from "three-spritetext"
interface Props {
  graphData: { nodes: any[]; links: any[] }
  width: number
  height: number
  communityColor: (c: number) => string
  linkColor: (l: any) => string
  nodeLabel: (n: any) => string
  linkLabel: (l: any) => string
  getRef: (instance: any) => void
  onEngineStop: () => void
}
export default function ForceGraph3DView({
  graphData, width, height, communityColor, linkColor,
  nodeLabel, linkLabel, getRef, onEngineStop,
}: Props) {
  return (
    <ForceGraph3D
      ref={(el: any) => getRef(el)}
      width={width}
      height={height}
      graphData={graphData}
      backgroundColor="rgba(0,0,0,0)"
      onEngineStop={onEngineStop}
      nodeLabel={nodeLabel}
      linkLabel={linkLabel}
      linkColor={linkColor}
      linkWidth={1}
      nodeColor={(n: any) => communityColor(n.community)}
      nodeThreeObjectExtend
      nodeThreeObject={(node: any) => {
        const sprite = new SpriteText(node.id)
        sprite.color = "#ffffff"
        sprite.textHeight = 4
        sprite.position.set(0, 6, 0)
        return sprite
      }}
    />
  )
}