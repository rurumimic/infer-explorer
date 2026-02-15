<script>
  import { getRuntime, runtimeStatus } from '../../lib/runtime';
  import { get } from 'svelte/store';

  let runRequestText = '';
  let referenceText = '';
  let browserResult = null;
  let report = null;

  async function runCompare() {
    const runtime = await getRuntime();
    const req = runRequestText ? JSON.parse(runRequestText) : get(runtimeStatus).latestRunRequest;
    const reference = JSON.parse(referenceText);
    browserResult = runtime.run_request(req);
    report = runtime.compare_results(browserResult, reference);
  }
</script>

<h2 class="mb-3 text-xl font-semibold">Compare</h2>
<div class="grid gap-4 md:grid-cols-2">
  <div class="card">
    <h3 class="mb-2 font-semibold">RunRequest JSON</h3>
    <textarea class="h-60 w-full bg-slate-950 text-xs" bind:value={runRequestText} placeholder="Playground에서 생성했거나 직접 붙여넣기"></textarea>
  </div>
  <div class="card">
    <h3 class="mb-2 font-semibold">Reference RunResult JSON</h3>
    <textarea class="h-60 w-full bg-slate-950 text-xs" bind:value={referenceText}></textarea>
  </div>
</div>
<button class="mt-3 rounded bg-indigo-600 px-3 py-1" on:click={runCompare}>Run in Browser (from encoded)</button>

{#if report}
  <div class="card mt-4">
    <h3 class="mb-2 font-semibold">Diff Metrics</h3>
    <pre class="text-xs">{JSON.stringify(report, null, 2)}</pre>
  </div>
{/if}

{#if browserResult}
  <div class="card mt-4">
    <h3 class="mb-2 font-semibold">Browser RunResult</h3>
    <pre class="text-xs">{JSON.stringify(browserResult, null, 2)}</pre>
  </div>
{/if}
