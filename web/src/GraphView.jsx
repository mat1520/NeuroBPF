import { useEffect, useMemo, useRef, useState } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import ForceGraph3D from 'react-force-graph-3d';
import * as THREE from 'three';

const EDGE_COLORS = {
  EXEC: '#7dd3fc',
  READ: '#4ade80',
  WRITE: '#facc15',
  UNLINK: '#fb923c',
  CONNECT: '#818cf8',
  ACCEPT: '#c084fc',
};

const NODE_KINDS = {
  process: { color: '#22d3ee', label: 'proceso' },
  file: { color: '#a3e635', label: 'archivo' },
  net: { color: '#f472b6', label: 'red' },
};

export const NODE_COLOR = '#22d3ee';
export const ANOM_COLOR = '#f43f5e';

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener('change', update);
    return () => mq.removeEventListener('change', update);
  }, []);
  return reduced;
}

function nodeLabel(node) {
  return node.comm || node.path || node.addr || node.label || node.id;
}

function nodeRadius(node) {
  const score = typeof node.score === 'number' ? node.score : 0;
  return 4 + score * 6;
}

function nodeColor(node) {
  if (node.anomalous) return ANOM_COLOR;
  const kind = NODE_KINDS[node.kind] || NODE_KINDS.process;
  return kind.color;
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

function roundRectPath(ctx, x, y, w, h) {
  const r = Math.min(w, h) / 3;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function diamondPath(ctx, x, y, r) {
  ctx.beginPath();
  ctx.moveTo(x, y - r);
  ctx.lineTo(x + r, y);
  ctx.lineTo(x, y + r);
  ctx.lineTo(x - r, y);
  ctx.closePath();
}

function draw2dNode(node, ctx, globalScale) {
  const size = nodeRadius(node);
  const scale = 1 / globalScale;
  const color = nodeColor(node);
  const anomalous = node.anomalous;

  const pulse = anomalous ? 1 + 0.18 * Math.abs(Math.sin(Date.now() / 300)) : 1;
  const r = size * pulse;

  ctx.save();
  if (anomalous) {
    ctx.beginPath();
    ctx.arc(node.x, node.y, r + 5 * scale, 0, 2 * Math.PI);
    ctx.fillStyle = 'rgba(244,63,94,0.15)';
    ctx.fill();
  }

  ctx.fillStyle = color;
  switch (node.kind) {
    case 'file': {
      const w = r * 1.5;
      const h = r * 1.5;
      roundRectPath(ctx, node.x - w / 2, node.y - h / 2, w, h);
      break;
    }
    case 'net':
      diamondPath(ctx, node.x, node.y, r * 1.3);
      break;
    default:
      ctx.beginPath();
      ctx.arc(node.x, node.y, r, 0, 2 * Math.PI);
  }
  ctx.fill();

  ctx.strokeStyle = anomalous ? 'rgba(255,255,255,0.85)' : 'rgba(2,6,23,0.6)';
  ctx.lineWidth = 1.2 * scale;
  ctx.stroke();

  const fontSize = 11 * scale;
  ctx.font = '500 11px "Fira Code", monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  ctx.fillStyle = anomalous ? '#fda4af' : 'rgba(226,232,240,0.85)';
  ctx.fillText(node.displayLabel, node.x, node.y + r + 4 * scale);
  ctx.restore();
}

function node3dObject(node) {
  const r = nodeRadius(node);
  const material = new THREE.MeshStandardMaterial({
    color: nodeColor(node),
    emissive: node.anomalous ? '#f43f5e' : nodeColor(node),
    emissiveIntensity: node.anomalous ? 0.35 : 0.12,
    roughness: 0.35,
    metalness: 0.1,
  });

  let geometry;
  if (node.kind === 'file') geometry = new THREE.BoxGeometry(r * 1.6, r * 1.6, r * 1.6);
  else if (node.kind === 'net') geometry = new THREE.OctahedronGeometry(r * 1.5);
  else geometry = new THREE.SphereGeometry(r, 20, 20);

  const mesh = new THREE.Mesh(geometry, material);
  const group = new THREE.Group();
  group.add(mesh);

  const label = textSprite(node.displayLabel, node.anomalous);
  label.position.y = r * 1.6 + 2;
  group.add(label);
  return group;
}

function textSprite(text, anomalous) {
  const canvas = document.createElement('canvas');
  const fontSize = 48;
  const measure = canvas.getContext('2d');
  measure.font = `500 ${fontSize}px "Fira Code", monospace`;
  const width = measure.measureText(text).width + 32;
  canvas.width = Math.max(64, Math.ceil(width));
  canvas.height = fontSize + 24;

  const ctx = canvas.getContext('2d');
  ctx.font = `500 ${fontSize}px "Fira Code", monospace`;
  ctx.fillStyle = anomalous ? '#fda4af' : '#e2e8f0';
  ctx.fillText(text, 16, fontSize);

  const texture = new THREE.CanvasTexture(canvas);
  texture.anisotropy = 4;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
  sprite.scale.set(canvas.width / 60, canvas.height / 60, 1);
  return sprite;
}

export default function GraphView({ frame, mode, onSelect, selected }) {
  const graphData = useMemo(() => toGraphData(frame), [frame]);
  const graphRef = useRef();
  const reduced = usePrefersReducedMotion();
  const selectedId = selected?.id;

  useEffect(() => {
    if (mode !== '2d') return undefined;
    let raf;
    const tick = () => {
      const graph = graphRef.current;
      if (graph && typeof graph.refresh === 'function') graph.refresh();
      raf = requestAnimationFrame(tick);
    };
    if (reduced) return undefined;
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mode, reduced]);

  const commonProps = {
    graphData,
    linkColor: (l) => (selectedId && (l.source.id === selectedId || l.target.id === selectedId) ? '#fbbf24' : l.color),
    linkOpacity: 0.5,
    linkWidth: (l) => (selectedId && (l.source.id === selectedId || l.target.id === selectedId) ? 2.2 : 1),
    onNodeClick: (node) => onSelect(node),
  };

  if (!frame) {
    return <div className="graph-blank" aria-hidden="true" />;
  }

  if (mode === '3d') {
    return (
      <ForceGraph3D
        ref={graphRef}
        {...commonProps}
        backgroundColor="#020617"
        nodeThreeObject={node3dObject}
        linkDirectionalArrowLength={4}
        linkDirectionalArrowRelPos={1}
        linkDirectionalParticles={(l) => (selectedId && (l.source.id === selectedId || l.target.id === selectedId) ? 2 : 0)}
      />
    );
  }

  return (
    <ForceGraph2D
      ref={graphRef}
      {...commonProps}
      nodeCanvasObject={draw2dNode}
      linkDirectionalArrowLength={3.5}
      linkDirectionalArrowRelPos={1}
      linkDirectionalParticles={(l) => (selectedId && (l.source.id === selectedId || l.target.id === selectedId) ? 2 : 0)}
    />
  );
}