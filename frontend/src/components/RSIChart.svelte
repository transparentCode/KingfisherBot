<script>
  import LightweightChart from './LightweightChart.svelte';
  import { marketStore } from '../stores/marketStore';
  import { assetStore } from '../stores/assetStore';

  export let visible = false;

  let loading = false;
  let error = null;
  let tvlcData = { candles: [], lines: [], markers: [], histograms: [] };

  // Default params matching RSI route
  let params = {
    length: 14,
    source: 'close',
    fc_enabled: true,
    fc_lookback: 50,
    fc_mult: 2.0,
    fc_zigzag_dev: 0.05
  };

  $: if (visible && $assetStore.selectedAsset && $marketStore.startDateTime && $marketStore.endDateTime) {
    fetchRSIData();
  }

  async function fetchRSIData() {
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

      const res = await fetch(`/api/rsi/calculate?${queryParams}`);
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
          main: incoming.main || [],
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
    <h3>RSI Analysis</h3>
    <div class="controls">
       <button class="icon-btn" on:click={fetchRSIData} title="Refresh">↻</button>
    </div>
  </div>
  
  {#if error}
    <div class="error">{error}</div>
  {:else}
    <div class="chart-container" class:loading-overlay={loading}>
      <LightweightChart 
        {tvlcData} 
        height={520} 
        chartType="line" 
        mainSeriesOptions={{ color: '#F2E94E', lineWidth: 1 }} 
      />
      {#if loading}
        <div class="spinner">Updating...</div>
      {/if}
    </div>
  {/if}
</div>
{/if}

<style>
  .card {
    background: #1e222d;
    border: 1px solid #2B2B43;
    border-radius: 4px;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
  }
  .card-header {
    padding: 0.5rem 1rem;
    border-bottom: 1px solid #2B2B43;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  h3 {
    margin: 0;
    font-size: 1rem;
    color: #d1d4dc;
  }
  .chart-container {
    position: relative;
    flex: 1;
    min-height: 520px;
  }
  .error {
    color: #ef5350;
    padding: 1rem;
  }
  .controls {
    display: flex;
    gap: 0.5rem;
  }
  .icon-btn {
    background: none;
    border: none;
    color: #787b86;
    cursor: pointer;
    font-size: 1.2rem;
    padding: 0 0.5rem;
  }
  .icon-btn:hover {
    color: #d1d4dc;
  }
  .loading-overlay {
    opacity: 0.7;
    pointer-events: none;
  }
  .spinner {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #d1d4dc;
    font-size: 1.2rem;
    background: rgba(30, 34, 45, 0.8);
    padding: 1rem 2rem;
    border-radius: 4px;
    z-index: 10;
  }
</style>