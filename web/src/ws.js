export function connectSnapshots(url) {
  const queue = [];
  const waiters = [];
  let closed = false;

  function push(frame) {
    if (closed) return;
    if (frame && frame.done === true) return;
    const waiter = waiters.shift();
    if (waiter) {
      waiter(frame);
    } else {
      queue.push(frame);
    }
  }

  const ws = new globalThis.WebSocket(url);
  ws.onmessage = (evt) => {
    if (closed) return;
    let frame;
    try {
      frame = JSON.parse(evt.data);
    } catch {
      return;
    }
    push(frame);
  };

  return {
    async next() {
      if (queue.length) return { value: queue.shift(), done: false };
      if (closed) return { value: undefined, done: true };
      return new Promise((resolve) => {
        waiters.push((frame) =>
          resolve(frame === undefined ? { value: undefined, done: true } : { value: frame, done: false }),
        );
      });
    },
    async close() {
      closed = true;
      const pending = waiters.splice(0);
      for (const waiter of pending) waiter();
      ws.close();
    },
    _push: push,
  };
}