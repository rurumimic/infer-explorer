<script>
  import { getRuntime } from '../../lib/runtime';

  let text = 'hello world';
  let query = 'query';
  let doc = 'document';
  let maxLength = 16;
  let encoded = null;

  async function runSingle() {
    const runtime = await getRuntime();
    encoded = runtime.encode_texts([text], maxLength);
  }

  async function runPair() {
    const runtime = await getRuntime();
    encoded = runtime.encode_pairs(query, [doc], maxLength);
  }
</script>

<h2 class="mb-3 text-xl font-semibold">Tokenizer Inspector</h2>
<div class="grid gap-4 md:grid-cols-2">
  <div class="card space-y-2">
    <h3 class="font-semibold">Single</h3>
    <textarea class="w-full bg-slate-950" bind:value={text}></textarea>
    <button class="rounded bg-indigo-600 px-3 py-1" on:click={runSingle}>Encode</button>
  </div>
  <div class="card space-y-2">
    <h3 class="font-semibold">Pair(query, doc)</h3>
    <input class="w-full bg-slate-950" bind:value={query} />
    <textarea class="w-full bg-slate-950" bind:value={doc}></textarea>
    <input type="number" bind:value={maxLength} />
    <button class="rounded bg-indigo-600 px-3 py-1" on:click={runPair}>Encode pair</button>
  </div>
</div>

{#if encoded}
  <div class="card mt-4">
    <button class="mb-2 rounded bg-slate-700 px-3 py-1" on:click={() => navigator.clipboard.writeText(JSON.stringify(encoded, null, 2))}>encoded JSON export</button>
    <pre class="overflow-auto text-xs">{JSON.stringify(encoded, null, 2)}</pre>
  </div>
{/if}
