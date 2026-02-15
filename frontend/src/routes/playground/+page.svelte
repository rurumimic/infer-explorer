<script>
  import { getRuntime, runtimeStatus } from '../../lib/runtime';
  import { get } from 'svelte/store';

  let task = 'embedding';
  let text = 'example text';
  let query = 'what is rust?';
  let docsRaw = 'rust is a language\npython is another';
  let result = null;

  async function run() {
    const runtime = await getRuntime();
    const docs = docsRaw.split('\n').filter(Boolean);
    const encoded = task === 'reranker' ? runtime.encode_pairs(query, docs, 32) : runtime.encode_texts([text], 32);
    const request = {
      schema_version: 1,
      run_id: crypto.randomUUID(),
      task,
      model: { arch: 'bert', root_prefix: 'auto', notes: null },
      tokenization: { max_length: 32, padding: 'max_length', truncation: 'longest_first' },
      options: { embedding_pooling: 'mean', embedding_normalize: 'none', top_k: 10, classifier_prob: 'softmax' },
      inputs: { texts: [text], query, docs },
      encoded
    };
    result = runtime.run_request(request);
    runtimeStatus.update((prev) => ({ ...prev, latestRunRequest: request, latestRunResult: result }));
  }

  function exportJson(data) {
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
  }
</script>

<h2 class="mb-3 text-xl font-semibold">Playground</h2>
<div class="card mb-4">
  <div class="mb-3 space-x-2">
    <button class="rounded bg-slate-700 px-2 py-1" on:click={() => (task = 'embedding')}>Embedding</button>
    <button class="rounded bg-slate-700 px-2 py-1" on:click={() => (task = 'classifier')}>Classifier</button>
    <button class="rounded bg-slate-700 px-2 py-1" on:click={() => (task = 'reranker')}>Reranker</button>
  </div>
  {#if task === 'reranker'}
    <input class="mb-2 w-full bg-slate-950" bind:value={query} />
    <textarea class="w-full bg-slate-950" rows="5" bind:value={docsRaw}></textarea>
  {:else}
    <textarea class="w-full bg-slate-950" rows="4" bind:value={text}></textarea>
  {/if}
  <button class="mt-2 rounded bg-indigo-600 px-3 py-1" on:click={run}>Run</button>
</div>

{#if result}
  <div class="card">
    <div class="mb-2 space-x-2">
      <button class="rounded bg-slate-700 px-2 py-1" on:click={() => exportJson(get(runtimeStatus).latestRunRequest)}>RunRequest JSON export</button>
      <button class="rounded bg-slate-700 px-2 py-1" on:click={() => exportJson(result)}>RunResult JSON export</button>
    </div>
    <pre class="overflow-auto text-xs">{JSON.stringify(result, null, 2)}</pre>
  </div>
{/if}
