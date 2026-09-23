// Layout extensions registered in components/graph/FlowGraph.tsx. Their
// published typings pull a second copy of cytoscape into the tree, so the
// one export used here is declared instead.
declare module "cytoscape-dagre" {
  const ext: import("cytoscape").Ext;
  export default ext;
}
declare module "cytoscape-fcose" {
  const ext: import("cytoscape").Ext;
  export default ext;
}
