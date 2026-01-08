<script>
  import LightweightChart from './LightweightChart.svelte';
  import { marketStore } from '../stores/marketStore';
  import { assetStore } from '../stores/assetStore';

  export let visible = false;

  let loading = false;
  let error = null;
  let tvlcData = { candles: [], lines: [], markers: [], histograms: [] };

  // Default params matching notebook
  let params = {
    lookback: 150,
    pivot_method: 'fractal',
    zigzag_dev: 0.05,
    pivot_window: 5,
    mode: 'geometric'
  };

  $: if (visible && $assetStore.selectedAsset) {
    fetchFractalData();
  }

  async function fetchFractalData() {
    loading = true;
    error = null;
    try {
      const symbol = $assetStore.selectedAsset || 'BTCUSDT';
      const timeframe = $marketStore.timeframe || '1h';
      const startDateTime = $marketStore.startDateTime;
      const endDateTime = $marketStore.endDateTime;

      const queryParams = new URLSearchParams({
        symbol,
        interval: timeframe,
        ...params
      });

      if (startDateTime) queryParams.set('startDateTime', startDateTime);
      if (endDateTime) queryParams.set('endDateTime', endDateTime);

      const res = await fetch(`/api/fractal/calculate?${queryParams}`);
      const data = await res.json();
      
        if (!data.success) {
          error = data.error;
          return;
        }

        // Prefer backend-provided TVLC data
        const incoming = data.tvlc_data || {};

        // Fallback: if backend didn’t send candles, use marketStore candles
        const fallbackCandles = $marketStore.candles.map(c => ({
          time: c.time,
          open: c.open,
          high: c.high,
          low: c.low,
          close: c.close,
          volume: c.volume
        }));

        tvlcData = {
          candles: incoming.candles && incoming.candles.length ? incoming.candles : fallbackCandles,
          lines: incoming.lines || [],
          markers: incoming.markers || [],
          histograms: incoming.histograms || []
        };
    } catch (e) {
      error = e.message;
    } finally {
      loading = false;
    }
  }
</script>

{#if visible}
<div class="card chart-card">
  <div class="card-header">
    <h3>Fractal Channel Analysis</h3>
    <div class="controls">
       <button class="icon-btn" on:click={fetchFractalData} title="Refresh">↻</button>
    </div>
  </div>
  
  {#if error}
    <div class="error">{error}</div>
  {:else}
    <div class="chart-container" class:loading-overlay={loading}>
      <LightweightChart {tvlcData} height={520} />
      {#if loading}
        <div class="spinner">Updating...</div>
      {/if}
    </div>
  {/if}
</div>
{/if}

<style>
  .loading { padding: 2rem; text-align: center; color: #888; }
  .error { padding: 2rem; text-align: center; color: #ef476f; }
  .controls { display: flex; gap: 0.5rem; }
  .chart-container { position: relative; }
  .loading-overlay { opacity: 0.6; pointer-events: none; }
  .spinner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: rgba(0,0,0,0.7);
    padding: 0.5rem 1rem;
    border-radius: 4px;
    color: white;
    z-index: 10;
  }
</style>
