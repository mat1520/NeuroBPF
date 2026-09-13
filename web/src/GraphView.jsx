import { useEffect, useMemo, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';

const EDGE_COLORS = {
  EXEC: '#94a3b8',
  READ: '#22c55e',
  WRITE: '#eab308',
  UNLINK: '#f97316',
  CONNECT: '#3b82f6',
  ACCEPT: '#8b5cf6',
};

export const NODE_COLOR = '#14b8a6';
export const ANOM_COLOR = '#ef4444';

function nodeLabel(node) {
  return node.comm || node.path || node.addr || node.label || node.id;
}

function nodeRadius(node) {
  const score = typeof node.score === 'number' ? node.score : 0;
  return Math.max(2, 2 + score * 10);
}

function toGraphData(frame) {
  if (!frame) return { nodes: [], links: [] };
  return {
    nodes: (frame.nodes || []).map((n) => ({ ...n, displayLabel: nodeLabel(n) })),
    links: (frame.edges || []).map((e) => ({
      source: e.src,
      target: e.dst,
      label: e.label,
      color: EDGE_COLORS[e.label] || '#64748b',
    })),
  };
}

function draw2dNode(node, ctx, globalScale) {
  const base = nodeRadius(node);
  const pulse = node.anomalous ? 1 + 0.25 * Math.sin(Date.now() / 400) : 1;
  const r = base * pulse;

  ctx.beginPath();
  ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
  ctx.fillStyle = node.anomalous ? ANOM_COLOR : NODE_COLOR;
  ctx.fill();

  if (node.anomalous) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, r + 3 / globalScale, 0, 2 * Math.PI);
    ctx.strokeStyle = ANOM_COLOR;
    ctx.globalAlpha = 0.6 - 0.4 * ((pulse - 1) / 0.25);
    ctx.lineWidth = 1.5 / globalScale;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  const fontSize = 12 / globalScale;
  ctx.font = `${fontSize}px system-ui, sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText(node.displayLabel, node.x, node.y + r + 4 / globalScale);
}

function textSprite(text) {
  const canvas = document.createElement('canvas');
  const fontSize = 40;
  const measure = canvas.getContext('2d');
  measure.font = `${fontSize}px system-ui, sans-serif`;
  const width = measure.measureText(text).width + 24;
  canvas.width = Math.ceil(width);
  canvas.height = fontSize + 16;

  const ctx = canvas.getContext('2d');
  ctx.font = `${fontSize}px system-ui, sans-serif`;
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText(text, 12, fontSize);

  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
  sprite.scale.set(canvas.width / 64, canvas.height / 64, 1);
  return sprite;
}

function node3dObject(node) {
  const r = nodeRadius(node);
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(r, 24, 24),
    new THREE.MeshStandardMaterial({ color: node.anomalous ? ANOM_COLOR : NODE_COLOR }),
  );
  const group = new THREE.Group();
  group.add(mesh);
  const label = textSprite(node.displayLabel);
  label.position.y = r + 2;
  group.add(label);
  return group;
}

export default function GraphView({ frame, mode }) {
  const graphData = useMemo(() => toGraphData(frame), [frame]);
  const graphRef = useRef();

  useEffect(() => {
    if (mode !== '2d') return undefined;
    let raf;
    const tick = () => {
      const graph = graphRef.current;
      if (graph && typeof graph.refresh === 'function') graph.refresh();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mode]);

  if (!frame) {
    return <div className="empty-state">waiting for snapshots...</div>;
  }

  if (mode === '3d') {
    return (
      <ForceGraph3D
        ref={graphRef}
        graphData={graphData}
        backgroundColor="#0f172a"
        nodeThreeObject={node3dObject}
        linkColor={(link) => link.color}
        linkOpacity={0.45}
      />
    );
  }

  return (
    <ForceGraph2D
      ref={graphRef}
      graphData={graphData}
      nodeCanvasObject={draw2dNode}
      linkColor={(link) => link.color}
      linkOpacity={0.45}
      linkDirectionalArrowLength={3}
      linkDirectionalArrowRelPos={1}
    />
  );
}