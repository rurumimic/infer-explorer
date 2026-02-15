<script>
  import { getRuntime, pushLog, setLoaded } from '../../lib/runtime';

  let configFile = null;
  let tokenizerFile = null;
  let modelFile = null;
  let diagnostics = null;

  async function loadModel() {
    if (!configFile || !tokenizerFile || !modelFile) {
      pushLog('필수 파일을 모두 선택하세요.');
      return;
    }
    const runtime = await getRuntime();
    diagnostics = runtime.load_model(
      Array.from(new Uint8Array(await configFile.arrayBuffer())),
      Array.from(new Uint8Array(await tokenizerFile.arrayBuffer())),
      Array.from(new Uint8Array(await modelFile.arrayBuffer())),
      undefined
    );
    setLoaded(true);
    pushLog('모델 로드 완료');
  }
</script>

<h2 class="mb-3 text-xl font-semibold">Model Loader</h2>
<div class="card space-y-3">
  <label class="block">config.json <input type="file" on:change={(e) => (configFile = e.target.files?.[0] ?? null)} /></label>
  <label class="block">tokenizer.json <input type="file" on:change={(e) => (tokenizerFile = e.target.files?.[0] ?? null)} /></label>
  <label class="block">model.safetensors <input type="file" on:change={(e) => (modelFile = e.target.files?.[0] ?? null)} /></label>
  <button class="rounded bg-indigo-600 px-3 py-1" on:click={loadModel}>Load</button>
</div>

{#if diagnostics}
  <div class="card mt-4">
    <h3 class="mb-2 font-semibold">Load diagnostics</h3>
    <pre class="overflow-auto text-xs">{JSON.stringify(diagnostics, null, 2)}</pre>
  </div>
{/if}
