import { writable } from 'svelte/store';

export const runtimeStatus = writable({
  loaded: false,
  webgpuReady: typeof navigator !== 'undefined' && 'gpu' in navigator,
  logs: []
});

let runtime;

export async function getRuntime() {
  if (runtime) {
    return runtime;
  }
  const mod = await import('./wasm-placeholder');
  runtime = new mod.WasmRuntime();
  return runtime;
}

export function pushLog(message) {
  runtimeStatus.update((prev) => ({ ...prev, logs: [message, ...prev.logs].slice(0, 20) }));
}

export function setLoaded(loaded) {
  runtimeStatus.update((prev) => ({ ...prev, loaded }));
}
