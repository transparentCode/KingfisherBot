<script>
  import { onMount, onDestroy } from 'svelte'
  import { marketStore } from '../stores/marketStore.js'
  import { assetStore } from '../stores/assetStore.js'
  import { indicatorStore } from '../stores/indicatorStore.js'
  import { authStore } from '../stores/authStore.js'
  import FractalChart from '../components/FractalChart.svelte'
  import MarketRegimeWidget from '../components/MarketRegimeWidget.svelte'
  import RSIChart from '../components/RSIChart.svelte'
  import LightweightChart from '../components/LightweightChart.svelte'

  let showFractalChart = true
  let showRSIChart = true
  let selectedTimeframe = '1h'
  const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
  const emptyTvlc = { candles: [], lines: [], markers: [], histograms: [] }

  // Date range controls
  const now = new Date()
  const defaultStart = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)
  let startDateTime = defaultStart.toISOString().slice(0, 16)
  let endDateTime = now.toISOString().slice(0, 16)
  let activeRangeLabel = '1M';

  const ranges = [
    { label: '1D', val: 1 },
    { label: '5D', val: 5 },
    { label: '1M', val: 30 },
    { label: '3M', val: 90 },
    { label: 'YTD', val: 'ytd' },
    { label: '1Y', val: 365 },
    { label: '5Y', val: 365 * 5 }
  ];

  function applyRange(range) {
      activeRangeLabel = range.label;
      const n = new Date();
      // Round to current minute to avoid seconds clutter in ISO
      n.setSeconds(0, 0); 
      endDateTime = n.toISOString().slice(0, 16);
      
      let s = new Date(n);
      if (range.val === 'ytd') {
          s = new Date(n.getFullYear(), 0, 1);
      } else {
          s.setTime(n.getTime() - range.val * 24 * 60 * 60 * 1000);
      }
      startDateTime = s.toISOString().slice(0, 16);
      
      // Allow UI to update bindings before fetching? No, Svelte is reactive locally.
      // But we just updated the variables.
      handleManualRefresh();
  }

  // React to asset/timeframe changes
  // We use a debounce or check to avoid initial double-fetch on mount if stores are already populated
  $: if ($authStore.token && $assetStore.selectedAsset) {
    refreshAllData();
  }

  // Ensure indicators are fetched if needed
  $: if ($assetStore.selectedAsset && $indicatorStore.availableIndicators.length > 0) {
      const needsFetch = $indicatorStore.availableIndicators.some(i => i.visible && !i.tvlcData && !i.loading);
      if (needsFetch) {
          indicatorStore.refreshVisibleIndicators();
      }
  }

  function formatISO(dateStr) {
      return new Date(dateStr).toISOString();
  }

  function fetchCandlesWithRange(symbol, timeframe = selectedTimeframe) {
    const sym = symbol || $assetStore.selectedAsset || 'BTCUSDT'
    marketStore.fetchCandles(sym, timeframe, formatISO(startDateTime), formatISO(endDateTime))
  }

  function refreshAllData() {
      if (!$assetStore.selectedAsset) return;
      fetchCandlesWithRange($assetStore.selectedAsset, selectedTimeframe);
      marketStore.fetchRegime($assetStore.selectedAsset, selectedTimeframe);
      // We might want to clear old indicator data to avoid mismatch visual
      // indicatorStore.clearData(); // Optional, depends on UX preference
      indicatorStore.refreshVisibleIndicators();
  }

  const handleDateChange = (which, value) => {
    activeRangeLabel = null;
    if (which === 'start') startDateTime = value
    if (which === 'end') endDateTime = value
  }

  const handleManualRefresh = () => {
    refreshAllData();
  }

  // Derived State for Charts
  $: visibleIndicators = $indicatorStore.availableIndicators.filter(ind => ind.visible && (ind.tvlcData || ind.data));
  const overlayIds = ['EMA', 'ExponentialMovingAverage', 'VolumeProfile', 'VWR'];
  
  $: overlayIndicators = visibleIndicators.filter(ind => overlayIds.includes(ind.id));
  $: panelIndicators = visibleIndicators.filter(ind => !overlayIds.includes(ind.id));
  $: baseTvlc = $marketStore.tvlcData || emptyTvlc;
  
  $: overlayLines = overlayIndicators.flatMap(ind => ind.tvlcData?.lines || []);
  $: overlayMarkers = overlayIndicators.flatMap(ind => ind.tvlcData?.markers || []);
  $: overlayHistograms = overlayIndicators.flatMap(ind => ind.tvlcData?.histograms || []);

  $: mainTvlcData = {
    candles: baseTvlc.candles || [],
    lines: overlayLines,
    markers: overlayMarkers,
    histograms: overlayHistograms
  };

  const defaultPanelHeight = 560;
  const mainChartBaseHeight = 520;
  
  $: indicatorPanels = panelIndicators.map(ind => ({
    id: ind.id,
    name: ind.name || ind.id,
    tvlcData: ind.tvlcData || emptyTvlc,
    height: defaultPanelHeight
  }));
</script>

<header class="dashboard-header">
  <div class="toolbar-row main">
    <div class="left-group">
      <!-- Asset Selector -->
      <div class="asset-combo">
         <span class="combo-label">Asset</span>
         <select 
          class="clean-select"
          value={$assetStore.selectedAsset} 
          on:change={(event) => assetStore.selectAsset(event.currentTarget.value)}
        >
          {#each $assetStore.assets as asset}
            <option value={asset}>{asset}</option>
          {/each}
        </select>
      </div>

      <div class="v-sep"></div>

      <!-- Compact Timeframe Selector -->
      <div class="tf-selector">
        {#each timeframes as tf}
          <button 
            class="tf-btn {selectedTimeframe === tf ? 'active' : ''}" 
            on:click={() => selectedTimeframe = tf}
          >
            {tf}
          </button>
        {/each}
      </div>
    </div>

    <!-- Status Widget -->
    <div class="center-group">
      <MarketRegimeWidget />
    </div>

    <div class="right-group">
       <button class="ghost compact">
         <span class="status-indicator live"></span> Live
       </button>
    </div>
  </div>

  <div class="toolbar-row secondary">
    <div class="time-controls">
      <div class="quick-ranges">
        {#each ranges as r}
          <button 
            class="range-btn {activeRangeLabel === r.label ? 'active' : ''}" 
            on:click={() => applyRange(r)}
          >
            {r.label}
          </button>
        {/each}
      </div>

      <div class="divider"></div>

      <div class="date-inputs">
        <div class="date-field" on:click={(e) => e.currentTarget.querySelector('input').showPicker()}>
          <svg class="field-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
          <div class="field-content">
            <span class="field-label">From</span>
            <input type="datetime-local" bind:value={startDateTime} on:change={(e) => handleDateChange('start', e.currentTarget.value)} />
          </div>
        </div>

        <span class="range-arrow">→</span>

        <div class="date-field" on:click={(e) => e.currentTarget.querySelector('input').showPicker()}>
          <svg class="field-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
          <div class="field-content">
            <span class="field-label">To</span>
            <input type="datetime-local" bind:value={endDateTime} on:change={(e) => handleDateChange('end', e.currentTarget.value)} />
          </div>
        </div>

        <button class="refresh-fab" on:click={handleManualRefresh} title="Refresh Data">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"></path><path d="M1 20v-6h6"></path><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 1 20.49 15"></path></svg>
        </button>
      </div>
    </div>
  </div>
</header>

<section class="grid">
  <div class="card chart-card">
    <div class="card-header">
      <h3>Price & EMA</h3>
      <div class="indicator-toggles">
        {#each $indicatorStore.availableIndicators as ind (ind.id)}
          <button 
            class="toggle-btn {ind.visible ? 'active' : ''}" 
            on:click={() => indicatorStore.toggleIndicator(ind.id)}
            title={ind.description}
          >
            <span class="status-dot"></span>
            {ind.name}
          </button>
        {/each}
        
        <button 
          class="toggle-btn {showFractalChart ? 'active' : ''}" 
          on:click={() => showFractalChart = !showFractalChart}
          title="Show Fractal Channel Analysis"
        >
          <span class="status-dot"></span>
          Fractal Channel
        </button>

        <button 
          class="toggle-btn {showRSIChart ? 'active' : ''}" 
          on:click={() => showRSIChart = !showRSIChart}
          title="Show RSI Analysis"
        >
          <span class="status-dot"></span>
          RSI Analysis
        </button>
      </div>
      <div class="chip">TVLC</div>
    </div>
    <LightweightChart tvlcData={mainTvlcData} height={mainChartBaseHeight} autoFitKey={$marketStore.symbol} />
  </div>

  <FractalChart visible={showFractalChart} />
  <RSIChart visible={showRSIChart} />

  {#each indicatorPanels as panel (panel.id)}
    <div class="card chart-card">
      <div class="card-header">
        <h3>{panel.name}</h3>
        <div class="chip">TVLC</div>
      </div>
      <LightweightChart tvlcData={panel.tvlcData} height={panel.height} autoFitKey={$marketStore.symbol} />
    </div>
  {/each}
</section>

<style>
  .dashboard-header {
    display: flex;
    flex-direction: column;
    gap: 8px;
    margin-bottom: 8px;
  }
  .toolbar-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
  }
  .toolbar-row.main {
    height: 48px;
  }
  .toolbar-row.secondary {
    justify-content: flex-start;
  }
  .left-group, .center-group, .right-group {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .asset-combo {
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  .combo-label {
    font-size: 9px;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 700;
    margin-bottom: 2px;
  }
  .clean-select {
    background: transparent;
    color: var(--text);
    border: none;
    font-size: 1.25rem;
    font-weight: 700;
    padding: 0;
    cursor: pointer;
    font-family: inherit;
    outline: none;
  }
  .clean-select:hover {
    color: var(--accent);
  }
  .v-sep {
    width: 1px;
    height: 24px;
    background: var(--border);
  }
  .tf-selector {
    display: flex;
    background: var(--surface-1);
    padding: 2px;
    border-radius: 8px;
    gap: 2px;
  }
  .tf-btn {
    background: transparent;
    border: none;
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .tf-btn:hover {
    color: var(--text);
    background: var(--surface-2);
  }
  .tf-btn.active {
    background: var(--surface-2);
    color: var(--accent);
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
  }
  .status-indicator {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--muted);
    display: inline-block;
    margin-right: 6px;
  }
  .status-indicator.live {
    background: var(--success);
    box-shadow: 0 0 8px rgba(6, 214, 160, 0.4);
  }
  .compact {
    padding: 6px 12px;
    font-size: 12px;
  }
  
  .indicator-toggles {
    display: flex;
    gap: 0.5rem;
    margin-left: 1rem;
    flex-wrap: wrap;
    flex: 1;
  }
  .asset-selector {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .time-controls {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .quick-ranges {
    display: flex;
    background: var(--surface-1);
    border-radius: 6px;
    padding: 2px;
    gap: 2px;
  }
  .range-btn {
    background: transparent;
    border: none;
    color: var(--muted);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    transition: all 0.2s;
  }
  .range-btn:hover {
    color: var(--text);
    background: var(--surface-2);
  }
  .range-btn.active {
    background: var(--surface-2);
    color: var(--accent);
    font-weight: 700;
  }
  .divider {
    width: 1px;
    height: 16px;
    background: var(--border);
  }
  .date-inputs {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .date-field {
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--surface-2);
    padding: 0 12px;
    border-radius: 8px;
    border: 1px solid transparent; /* Reserve for focus */
    cursor: pointer;
    transition: all 0.2s;
    height: 40px;
    min-width: 160px;
  }
  .date-field:hover {
    background: var(--surface-3);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    transform: translateY(-1px);
  }
  .date-field:focus-within {
    border-color: var(--accent);
    background: var(--surface-3);
  }
  .field-icon {
    color: var(--muted);
    opacity: 0.7;
    flex-shrink: 0;
  }
  .field-content {
    display: flex;
    flex-direction: column;
    justify-content: center;
    position: relative;
    flex: 1;
    overflow: hidden;
  }
  .field-label {
    font-size: 9px;
    color: var(--muted);
    text-transform: uppercase;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 3px;
    letter-spacing: 0.5px;
  }
  .date-field input {
    background: transparent;
    border: none;
    color: var(--text);
    font-size: 11px;
    font-family: 'Space Grotesk', monospace;
    font-weight: 500;
    padding: 0;
    margin: 0;
    width: 100%;
    outline: none;
    line-height: 1.2;
  }
  /* Hide calendar picker indicator in some browsers as we have custom icon trigger */
  .date-field input::-webkit-calendar-picker-indicator {
    background: transparent;
    opacity: 0;
    position: absolute;
    width: 100%;
    height: 100%;
    cursor: pointer;
    top: 0; 
    left: 0;
  }
  .range-arrow {
    font-size: 12px;
    color: var(--muted);
    opacity: 0.5;
  }
  .refresh-fab {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    
    background: var(--accent); /* Vibrant accent color */
    color: #09090b; /* Dark text for contrast */
    box-shadow: 0 4px 12px rgba(76, 201, 240, 0.3); /* Colored shadow */
    transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1); /* Bouncy spring-like */
  }
  .refresh-fab:hover {
    transform: translateY(-2px) scale(1.05);
    box-shadow: 0 8px 16px rgba(76, 201, 240, 0.5);
    filter: brightness(1.1);
  }
  .refresh-fab:active {
    transform: translateY(0) scale(0.95);
    box-shadow: 0 2px 4px rgba(76, 201, 240, 0.2);
  }
  .toggle-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: transparent;
    padding: 0.25rem 0.75rem;
    border-radius: 99px; /* Pill shape */
    font-size: 0.75rem;
    color: var(--muted, #a0a0a0);
    border: 1px solid var(--border, rgba(255, 255, 255, 0.1));
    cursor: pointer;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .toggle-btn:hover {
    background: var(--surface-2, rgba(255, 255, 255, 0.1));
    color: #fff;
    transform: translateY(-1px);
    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
  }
  .toggle-btn.active {
    background: rgba(6, 214, 160, 0.15); /* Use --success alpha */
    border-color: transparent;
    color: var(--success, #4caf50);
    font-weight: 600;
  }
  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: currentColor;
    opacity: 0.6;
    transition: transform 0.2s;
  }
  .toggle-btn.active .status-dot {
    background: currentColor;
    opacity: 1;
    transform: scale(1.2);
  }
</style>