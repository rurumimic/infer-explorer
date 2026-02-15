<script >
  import '../app.css';
  import { runtimeStatus } from '$lib/runtime';
</script>

<div class="flex min-h-screen">
  <aside class="w-60 border-r border-slate-800 p-4">
    <h1 class="mb-4 text-lg font-bold">Infer Explorer</h1>
    <nav class="space-y-2 text-sm">
      <a class="block rounded px-2 py-1 hover:bg-slate-800" href="/load">Load Model</a>
      <a class="block rounded px-2 py-1 hover:bg-slate-800" href="/tokenizer">Tokenizer</a>
      <a class="block rounded px-2 py-1 hover:bg-slate-800" href="/playground">Playground</a>
      <a class="block rounded px-2 py-1 hover:bg-slate-800" href="/compare">Compare</a>
    </nav>
  </aside>
  <main class="flex-1 p-4">
    <div class="mb-4 rounded border border-slate-800 bg-slate-900 px-3 py-2 text-xs">
      <span class="mr-4">Model: {$runtimeStatus.loaded ? 'Loaded' : 'Not loaded'}</span>
      <span class="mr-4">WebGPU: {$runtimeStatus.webgpuReady ? 'Ready' : 'Unavailable'}</span>
      <span>Latest log: {$runtimeStatus.logs[0] ?? 'n/a'}</span>
    </div>
    <slot />
  </main>
</div>
