import { describe, beforeEach, expect, test } from 'vitest';
import { connectSnapshots } from '../ws.js';

class FakeWebSocket {
  static instances = [];

  constructor(url) {
    this.url = url;
    this.onmessage = null;
    this.onopen = null;
    this.onclose = null;
    this.closed = false;
    FakeWebSocket.instances.push(this);
  }

  close() {
    this.closed = true;
  }

  emit(data) {
    const handler = this.onmessage;
    if (handler) handler({ data });
  }
}

describe('connectSnapshots', () => {
  beforeEach(() => {
    globalThis.WebSocket = FakeWebSocket;
    FakeWebSocket.instances = [];
  });

  test('yields frames received over the wire in order', async () => {
    const cursor = connectSnapshots('ws://x');
    const ws = FakeWebSocket.instances[0];
    const f1 = { run_id: 'run1', run_type: 'attack', snapshot_index: 0, snap_total: 5, nodes: [], edges: [] };
    const f2 = { run_id: 'run1', run_type: 'attack', snapshot_index: 1, snap_total: 5, nodes: [], edges: [] };

    ws.emit(JSON.stringify(f1));
    ws.emit(JSON.stringify(f2));

    expect(await cursor.next()).toEqual({ value: f1, done: false });
    expect(await cursor.next()).toEqual({ value: f2, done: false });
  });

  test('buffers frames while a next() is already waiting', async () => {
    const cursor = connectSnapshots('ws://x');
    const ws = FakeWebSocket.instances[0];
    const f1 = { run_id: 'run1', snapshot_index: 0, nodes: [], edges: [] };

    const pending = cursor.next();
    ws.emit(JSON.stringify(f1));

    expect(await pending).toEqual({ value: f1, done: false });
  });

  test('a done frame does not terminate the cursor', async () => {
    const cursor = connectSnapshots('ws://x');
    const ws = FakeWebSocket.instances[0];
    const f1 = { run_id: 'run2', snapshot_index: 0, nodes: [], edges: [] };

    ws.emit(JSON.stringify({ done: true }));
    ws.emit(JSON.stringify(f1));

    expect(await cursor.next()).toEqual({ value: f1, done: false });
  });

  test('_push dev hook drives the cursor with parsed frames', async () => {
    const cursor = connectSnapshots('ws://x');
    const f1 = { run_id: 'run3', snapshot_index: 2, nodes: [], edges: [] };
    const f2 = { run_id: 'run3', snapshot_index: 3, nodes: [], edges: [] };

    cursor._push(f1);
    cursor._push({ done: true });
    cursor._push(f2);

    expect(await cursor.next()).toEqual({ value: f1, done: false });
    expect(await cursor.next()).toEqual({ value: f2, done: false });
  });

  test('close() resolves a pending next() as done', async () => {
    const cursor = connectSnapshots('ws://x');
    const ws = FakeWebSocket.instances[0];

    const pending = cursor.next();
    await cursor.close();

    expect(await pending).toEqual({ value: undefined, done: true });
    expect(ws.closed).toBe(true);
  });

  test('next() stays ordered when pushing in bursts via onmessage', async () => {
    const cursor = connectSnapshots('ws://x');
    const ws = FakeWebSocket.instances[0];
    const frames = [0, 1, 2, 3].map((i) => ({
      run_id: 'run4',
      snapshot_index: i,
      nodes: [{ id: `p:${i}`, label: 'bash', score: 0.5, anomalous: false }],
      edges: [],
    }));

    const pending = Promise.all([cursor.next(), cursor.next(), cursor.next(), cursor.next()]);
    frames.forEach((f, i) => ws.emit(JSON.stringify(f)));

    const results = await pending;
    results.forEach((r, i) => expect(r).toEqual({ value: frames[i], done: false }));
  });
});