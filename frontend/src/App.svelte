<script>
  import { onMount } from 'svelte'
  import { authStore } from './stores/authStore.js'
  import { marketStore } from './stores/marketStore.js'
  import { assetStore } from './stores/assetStore.js'
  import { configStore } from './stores/configStore.js'
  import { indicatorStore } from './stores/indicatorStore.js'
  import FractalChart from './components/FractalChart.svelte'
  import RSIChart from './components/RSIChart.svelte'
  import LightweightChart from './components/LightweightChart.svelte'
  import LogView from './components/LogView.svelte'
  import SystemMetricsView from './components/SystemMetricsView.svelte'
  import './modal.css'

  let username = ''
  let password = ''
  let showConfigModal = false
  let showFractalChart = true // Default visible as requested
  let showRSIChart = true // Default visible
  let activeTab = 'dashboard' // dashboard, strategies, backtests, logs
  let selectedTimeframe = '1h'
  const timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
  const emptyTvlc = { candles: [], lines: [], markers: [], histograms: [] }

  // Date range controls (defaults: last 30 days ending now)
  const now = new Date()
  const defaultStart = new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000)
  let startDateTime = defaultStart.toISOString().slice(0, 16) // for datetime-local input
  let endDateTime = now.toISOString().slice(0, 16)

  onMount(() => {
    if ($authStore.token) {
      assetStore.fetchAssets()
      indicatorStore.fetchAvailableIndicators()
    }
  })

  // React to auth changes
  $: if ($authStore.token) {
    assetStore.fetchAssets()
    indicatorStore.fetchAvailableIndicators()
  }

  // React to asset/timeframe changes and fetch candles once per change
  $: if ($authStore.token && $assetStore.selectedAsset) {
    fetchCandlesWithRange($assetStore.selectedAsset, selectedTimeframe)
    indicatorStore.clearData()
    indicatorStore.refreshVisibleIndicators()
  }

  function fetchCandlesWithRange(symbol, timeframe = selectedTimeframe) {
    const sym = symbol || $assetStore.selectedAsset || 'BTCUSDT'
    const startISO = new Date(startDateTime).toISOString()
    const endISO = new Date(endDateTime).toISOString()
    marketStore.fetchCandles(sym, timeframe, startISO, endISO)
  }

  // Ensure we fetch data for default visible indicators (like EMA) if they load after the asset
  $: if ($assetStore.selectedAsset && $indicatorStore.availableIndicators.length > 0) {
      const needsFetch = $indicatorStore.availableIndicators.some(i => i.visible && !i.data && !i.loading);
      if (needsFetch) {
          indicatorStore.refreshVisibleIndicators();
      }
  }

  const handleLogin = async (e) => {
    e.preventDefault()
    await authStore.login(username.trim(), password)
  }

  const handleDateChange = (which, value) => {
    if (which === 'start') startDateTime = value
    if (which === 'end') endDateTime = value
    // Manual refresh button will trigger fetch; no auto-fetch on date change
  }

  const handleManualRefresh = () => {
    if ($authStore.token && $assetStore.selectedAsset) {
      fetchCandlesWithRange($assetStore.selectedAsset, selectedTimeframe)
      indicatorStore.clearData()
      indicatorStore.refreshVisibleIndicators()
    }
  }

  const handleRefreshConfig = async () => {
    await configStore.reloadConfig()
  }

  const openConfig = async () => {
    await configStore.fetchConfig()
    showConfigModal = true
  }

  $: visibleIndicators = $indicatorStore.availableIndicators.filter(ind => ind.visible && (ind.tvlcData || ind.data));

  const overlayIds = ['EMA', 'ExponentialMovingAverage', 'VolumeProfile', 'VWR'];

  $: overlayIndicators = visibleIndicators.filter(ind => overlayIds.includes(ind.id));
  $: panelIndicators = visibleIndicators.filter(ind => !overlayIds.includes(ind.id));
  $: baseTvlc = $marketStore.tvlcData || emptyTvlc;
  $: overlayLines = overlayIndicators.flatMap(ind => ind.tvlcData?.lines || []);
  $: overlayMarkers = overlayIndicators.flatMap(ind => ind.tvlcData?.markers || []);
  $: overlayHistograms = overlayIndicators.flatMap(ind => ind.tvlcData?.histograms || []);

  // Main chart merges price candles with overlay indicator primitives
  $: mainTvlcData = {
    candles: baseTvlc.candles || [],
    lines: overlayLines,
    markers: overlayMarkers,
    histograms: overlayHistograms
  };

  // Individual panels per indicator (non-EMA)
  const defaultPanelHeight = 560;
  $: indicatorPanels = panelIndicators.map(ind => ({
    id: ind.id,
    name: ind.name || ind.id,
    tvlcData: ind.tvlcData || emptyTvlc,
    height: defaultPanelHeight
  }));

  const mainChartBaseHeight = 520;
</script>

{#if showConfigModal}
  <div class="modal-backdrop" role="button" tabindex="0" on:click|self={() => showConfigModal = false} on:keydown={(e) => (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') && (showConfigModal = false)}>
    <div class="modal-card">
      <header>
        <h2>System Configuration</h2>
        <button class="icon-btn" on:click={() => showConfigModal = false}>✕</button>
      </header>
      <div class="modal-body">
        {#if $configStore.loading}
          <p>Loading configuration...</p>
        {:else if $configStore.error}
          <p class="error">{$configStore.error}</p>
        {:else if $configStore.config}
          <pre>{JSON.stringify($configStore.config, null, 2)}</pre>
        {/if}
      </div>
      <footer>
        <button class="secondary" on:click={handleRefreshConfig} disabled={$configStore.loading}>
          {$configStore.loading ? 'Reloading...' : 'Reload from Disk'}
        </button>
        <button class="primary" on:click={() => showConfigModal = false}>Close</button>
      </footer>
    </div>
  </div>
{/if}

{#if !$authStore.token}
  <div class="auth-shell">
    <div class="lottie-bg">
        <!-- Remote JSON Lottie Animation -->
        <lottie-player src="https://lottie.host/d917fc46-0bb8-4e2d-ae51-3cd966b29c0a/CUP85pnpEt.json" background="transparent" speed="1" style="width: 100%; height: 100%;" loop autoplay></lottie-player>
    </div>
    <div class="auth-card">
      <header>
        <p class="eyebrow">KingfisherBot</p>
        <h1>Sign In</h1>
        <p class="muted">Enter your credentials to continue.</p>
      </header>

      <form class="auth-form" on:submit|preventDefault={handleLogin}>
        <label>
          <span>Username</span>
          <input
            type="text"
            bind:value={username}
            autocomplete="username"
            placeholder="admin"
            required
          />
        </label>

        <label>
          <span>Password</span>
          <input
            type="password"
            bind:value={password}
            autocomplete="current-password"
            placeholder="••••••••"
            required
          />
        </label>

        {#if $authStore.error}
          <div class="error-chip">{$authStore.error}</div>
        {/if}

        <button class="primary" type="submit" disabled={$authStore.loading}>
          {$authStore.loading ? 'Signing in…' : 'Sign In'}
        </button>
      </form>
    </div>
  </div>
{:else}
  <div class="app-shell">
    <div class="dashboard-bg">
        <!-- Remote JSON Lottie Animation (subtle) -->
        <lottie-player src="https://lottie.host/d917fc46-0bb8-4e2d-ae51-3cd966b29c0a/CUP85pnpEt.json" background="transparent" speed="0.5" style="width: 100%; height: 100%;" loop autoplay></lottie-player>
    </div>
    
    <header class="navbar">
      <div class="brand">
        <div class="dot live"></div>
        <span>KingfisherBot</span>
      </div>

      <nav>
        <button class="nav-btn" class:active={activeTab === 'dashboard'} on:click={() => activeTab = 'dashboard'}>Dashboard</button>
        <button class="nav-btn" class:active={activeTab === 'strategies'} on:click={() => activeTab = 'strategies'}>Strategies</button>
        <button class="nav-btn" class:active={activeTab === 'backtests'} on:click={() => activeTab = 'backtests'}>Backtests</button>
        <button class="nav-btn" class:active={activeTab === 'system'} on:click={() => activeTab = 'system'}>System Metrics</button>
        <button class="nav-btn" class:active={activeTab === 'logs'} on:click={() => activeTab = 'logs'}>System Logs</button>
        <button class="nav-btn" on:click={openConfig}>Settings</button>
      </nav>

      <div class="user-actions">
        <span class="muted">👋 {$authStore.user?.username}</span>
        <button class="ghost" on:click={() => authStore.logout()}>Logout</button>
      </div>
    </header>

    <main class="content">
      {#if activeTab === 'dashboard'}
      <header class="topbar">
        <div class="asset-selector">
          <select 
            value={$assetStore.selectedAsset} 
            on:change={(event) => assetStore.selectAsset(event.currentTarget.value)}
          >
            {#each $assetStore.assets as asset}
              <option value={asset}>{asset}</option>
            {/each}
          </select>
          
          <select bind:value={selectedTimeframe}>
            {#each timeframes as tf}
              <option value={tf}>{tf}</option>
            {/each}
          </select>

          <div class="date-range">
            <label>
              <span>Start</span>
              <input type="datetime-local" bind:value={startDateTime} on:change={(event) => handleDateChange('start', event.currentTarget.value)} />
            </label>
            <label>
              <span>End</span>
              <input type="datetime-local" bind:value={endDateTime} on:change={(event) => handleDateChange('end', event.currentTarget.value)} />
            </label>
            <button class="secondary" on:click={handleManualRefresh}>Refresh</button>
          </div>
        </div>
        <div>
          <p class="eyebrow">Symbol</p>
          <h2>{$marketStore.symbol}</h2>
        </div>
        <div class="top-actions">
          <button class="ghost">Pause Live</button>
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
              
              <!-- Special Toggle for Fractal Channel -->
              <button 
                class="toggle-btn {showFractalChart ? 'active' : ''}" 
                on:click={() => showFractalChart = !showFractalChart}
                title="Show Fractal Channel Analysis"
              >
                <span class="status-dot"></span>
                Fractal Channel
              </button>

              <!-- Special Toggle for RSI Chart -->
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
          <LightweightChart tvlcData={mainTvlcData} height={mainChartBaseHeight} />
        </div>

        <!-- Dedicated Fractal Chart -->
        <FractalChart visible={showFractalChart} />
        
        <!-- Dedicated RSI Chart -->
        <RSIChart visible={showRSIChart} />

        {#each indicatorPanels as panel (panel.id)}
          <div class="card chart-card">
            <div class="card-header">
              <h3>{panel.name}</h3>
              <div class="chip">TVLC</div>
            </div>
            <LightweightChart tvlcData={panel.tvlcData} height={panel.height} />
          </div>
        {/each}


      </section>
      {:else if activeTab === 'logs'}
        <LogView />
      {:else if activeTab === 'system'}
        <SystemMetricsView />
      {:else}
        <div class="placeholder-page">
          <h2>{activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}</h2>
          <p>This section is under construction.</p>
        </div>
      {/if}
    </main>
  </div>
{/if}

<style>
  .indicator-toggles {
    display: flex;
    gap: 0.5rem;
    margin-left: 1rem;
    flex-wrap: wrap;
    flex: 1;
  }
  .date-range {
    display: flex;
    gap: 0.5rem;
    align-items: flex-end;
  }
  .date-range label {
    display: flex;
    flex-direction: column;
    font-size: 0.75rem;
    color: #a0a0a0;
  }
  .date-range input {
    background: #11151d;
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: #e8eef7;
    padding: 0.35rem 0.5rem;
    border-radius: 0.35rem;
    font-size: 0.85rem;
  }
  .toggle-btn {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(255, 255, 255, 0.05);
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.75rem;
    color: #a0a0a0;
    border: 1px solid rgba(255, 255, 255, 0.1);
    cursor: pointer;
    transition: all 0.2s;
  }
  .toggle-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #fff;
  }
  .toggle-btn.active {
    background: rgba(76, 175, 80, 0.15);
    border-color: rgba(76, 175, 80, 0.3);
    color: #4caf50;
  }
  .status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #555;
    transition: background 0.2s;
  }
  .toggle-btn.active .status-dot {
    background: #4caf50;
  }

  /* Lottie Background Styles */
  .lottie-bg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    opacity: 0.4;
    pointer-events: none;
    overflow: hidden;
  }

  .dashboard-bg {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: 0;
    opacity: 0.4; /* Increased visibility further */
    pointer-events: none;
    overflow: hidden;
  }

  /* Ensure content sits above background */
  .auth-card {
    position: relative;
    z-index: 1;
    background: rgba(30, 30, 30, 0.85); /* Add some transparency/blur */
    backdrop-filter: blur(10px);
  }

  .content {
    position: relative;
    z-index: 1;
  }
</style>
