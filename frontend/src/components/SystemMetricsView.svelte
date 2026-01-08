<script>
  import { onMount, onDestroy } from 'svelte';
  import LightweightChart from './LightweightChart.svelte';

  let status = null;
  let loading = true;
  let error = null;
  let pollInterval;

  // Chart Data
  let cpuData = { main: [] };
  let msgRateData = { main: [] };
  let queueData = { main: [], lines: [] }; // main=write, lines=[calc]

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/system/status');
      if (res.ok) {
        status = await res.json();
      }
    } catch (e) {
      console.error('Failed to fetch system status', e);
    }
  };

  const fetchMetric = async (type) => {
    try {
      const res = await fetch(`/api/system/metrics?type=${type}&limit=1440`); // Last 24h (assuming 1/min)
      const json = await res.json();
      if (json.success) {
        return json.data.map(d => ({
          time: d.ts / 1000, // Convert ms to seconds for LightweightCharts
          value: d.val
        }));
      }
    } catch (e) {
      console.error(`Failed to fetch metric ${type}`, e);
    }
    return [];
  };

  const fetchAllMetrics = async () => {
    const [cpu, rate, qWrite, qCalc] = await Promise.all([
      fetchMetric('cpu_load'),
      fetchMetric('msg_rate'),
      fetchMetric('queue_write'),
      fetchMetric('queue_calc')
    ]);

    cpuData = { main: cpu };
    msgRateData = { main: rate };
    
    // For queues, we'll put Write in main and Calc in lines
    queueData = {
      main: qWrite,
      lines: [{
        name: 'Calc Queue',
        color: '#f4a261', // Orange
        data: qCalc
      }]
    };
  };

  const refresh = async () => {
    await fetchStatus();
    await fetchAllMetrics();
    loading = false;
  };

  onMount(() => {
    refresh();
    pollInterval = setInterval(refresh, 5000); // Poll every 5s
  });

  onDestroy(() => {
    if (pollInterval) clearInterval(pollInterval);
  });

  // Helpers
  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };
</script>

<div class="system-view">
  <header class="view-header">
    <h2>System Monitor</h2>
    <div class="status-badge {status?.status === 'online' ? 'online' : 'offline'}">
      {status?.status || 'Unknown'}
    </div>
  </header>

  {#if loading && !status}
    <div class="loading">Loading system metrics...</div>
  {:else}
    <!-- Status Cards -->
    <div class="grid-cards">
      <!-- Component Health -->
      <div class="card">
        <h3>Components</h3>
        <div class="stat-row">
          <span>Database</span>
          <span class="status-text {status?.components?.database === 'connected' ? 'ok' : 'err'}">
            {status?.components?.database}
          </span>
        </div>
        <div class="stat-row">
          <span>Redis</span>
          <span class="status-text {status?.components?.redis === 'connected' ? 'ok' : 'err'}">
            {status?.components?.redis}
          </span>
        </div>
        <div class="stat-row">
          <span>Workers</span>
          <span class="status-text {status?.components?.workers === 'healthy' ? 'ok' : 'warn'}">
            {status?.components?.workers || 'Unknown'}
          </span>
        </div>
      </div>

      <!-- Resources -->
      <div class="card">
        <h3>Resources</h3>
        <div class="stat-row">
          <span>Load Avg (1m)</span>
          <span class="value">{status?.resources?.load_avg?.[0]?.toFixed(2) || '-'}</span>
        </div>
        <div class="stat-row">
          <span>Disk Usage</span>
          <span class="value">{status?.resources?.disk?.percent}%</span>
        </div>
        <div class="progress-bar">
          <div class="fill" style="width: {status?.resources?.disk?.percent}%"></div>
        </div>
        <div class="sub-text">
          Free: {status?.resources?.disk?.free_gb} GB
        </div>
      </div>

      <!-- Market Service -->
      <div class="card">
        <h3>Market Service</h3>
        <div class="stat-row">
          <span>Assets</span>
          <span class="value">{status?.market_service?.configuration?.enabled_assets || 0} / {status?.market_service?.configuration?.total_assets || 0}</span>
        </div>
        <div class="stat-row">
          <span>Regime Adaptation</span>
          <span class="value">{status?.market_service?.configuration?.regime_adaptation ? 'ON' : 'OFF'}</span>
        </div>
        <div class="stat-row">
          <span>DB Writers</span>
          <span class="value">{status?.market_service?.workers?.db_writers?.length || 0}</span>
        </div>
      </div>
    </div>

    <!-- Charts -->
    <div class="charts-section">
      <div class="chart-card">
        <h3>CPU Load (1m)</h3>
        <div class="chart-wrapper">
          <LightweightChart 
            chartType="line" 
            tvlcData={cpuData} 
            height={200}
            mainSeriesOptions={{ color: '#4cc9f0', lineWidth: 2 }}
          />
        </div>
      </div>

      <div class="chart-card">
        <h3>Message Rate (msg/sec)</h3>
        <div class="chart-wrapper">
          <LightweightChart 
            chartType="line" 
            tvlcData={msgRateData} 
            height={200}
            mainSeriesOptions={{ color: '#4caf50', lineWidth: 2 }}
          />
        </div>
      </div>

      <div class="chart-card full-width">
        <h3>Queue Depth</h3>
        <div class="chart-wrapper">
          <LightweightChart 
            chartType="line" 
            tvlcData={queueData} 
            height={250}
            mainSeriesOptions={{ title: 'Write Queue', color: '#ef476f', lineWidth: 2 }}
          />
        </div>
        <div class="legend">
          <span style="color: #ef476f">● Write Queue</span>
          <span style="color: #f4a261">● Calc Queue</span>
        </div>
      </div>
    </div>
  {/if}
</div>

<style>
  .system-view {
    display: flex;
    flex-direction: column;
    gap: 20px;
    padding-bottom: 40px;
  }

  .view-header {
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .status-badge {
    padding: 4px 12px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
  }
  .status-badge.online { background: rgba(76, 175, 80, 0.2); color: #4caf50; border: 1px solid rgba(76, 175, 80, 0.4); }
  .status-badge.offline { background: rgba(239, 71, 111, 0.2); color: #ef476f; border: 1px solid rgba(239, 71, 111, 0.4); }

  .grid-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px;
  }

  .card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
  }

  .card h3 {
    margin: 0 0 16px 0;
    font-size: 1rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  .stat-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
    font-size: 0.95rem;
  }

  .status-text.ok { color: #4caf50; }
  .status-text.warn { color: #f4a261; }
  .status-text.err { color: #ef476f; }
  .value { font-weight: 600; font-family: monospace; }

  .progress-bar {
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    overflow: hidden;
    margin-top: 8px;
  }
  .fill {
    height: 100%;
    background: var(--accent);
  }

  .sub-text {
    font-size: 0.8rem;
    color: var(--muted);
    margin-top: 6px;
    text-align: right;
  }

  .charts-section {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 16px;
  }

  .chart-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    min-height: 260px;
  }

  .chart-card.full-width {
    grid-column: 1 / -1;
  }

  .chart-wrapper {
    margin-top: 12px;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }

  .legend {
    display: flex;
    gap: 16px;
    margin-top: 8px;
    font-size: 0.85rem;
    justify-content: center;
  }
</style>