<script>
  import { onMount, onDestroy } from 'svelte';
  import { createChart, CandlestickSeries, LineSeries, HistogramSeries, createSeriesMarkers } from 'lightweight-charts';

  export let tvlcData = { candles: [], lines: [], markers: [], histograms: [] };
  export let height = 520;
  export let options = {};
  export let chartType = 'candlestick'; // 'candlestick' or 'line'
  export let mainSeriesOptions = {};
  export let autoFitKey = null; // Change this to trigger auto-fit (e.g. Symbol change)

  let container;
  let legendContainer;
  let chart;
  let mainSeries; // Renamed from candleSeries
  let mainSeriesMarkers; // Renamed from candleSeriesMarkers
  let hasFitContent = false;
  let lastAutoFitKey = null;

  const lineSeriesMap = new Map();
  const histSeriesMap = new Map();

  $: if (autoFitKey !== lastAutoFitKey) {
      hasFitContent = false;
      lastAutoFitKey = autoFitKey;
  }

  const baseOptions = {
    layout: { background: { color: '#161a21' }, textColor: '#e8eef7' }, /* Match --surface-1 and --text */
    grid: { vertLines: { color: 'rgba(255, 255, 255, 0.05)' }, horzLines: { color: 'rgba(255, 255, 255, 0.05)' } },
    timeScale: { timeVisible: true, secondsVisible: false, borderColor: 'rgba(255, 255, 255, 0.1)' },
    rightPriceScale: { borderVisible: false },
    crosshair: { mode: 1 }
  };

  function initChart() {
    if (!container) return;
    chart = createChart(container, { width: container.clientWidth, height, ...baseOptions, ...options });
    
    // Ensure chart is fully initialized before adding series
    if (chart) {
        if (chartType === 'line') {
            mainSeries = chart.addSeries(LineSeries, mainSeriesOptions);
        } else {
            mainSeries = chart.addSeries(CandlestickSeries, mainSeriesOptions);
        }
        
        applyData(tvlcData);
        
        chart.subscribeCrosshairMove(param => {
            if (!legendContainer) return;
            
            if (
                param.point === undefined ||
                !param.time ||
                param.point.x < 0 ||
                param.point.x > container.clientWidth ||
                param.point.y < 0 ||
                param.point.y > container.clientHeight
            ) {
                // Optional: Clear legend or show last value
                // legendContainer.innerHTML = ''; 
                return;
            }

            let html = '';
            
            // Main Series Data
            const mainData = param.seriesData.get(mainSeries);
            if (mainData) {
                if (chartType === 'candlestick') {
                const main = /** @type {any} */ (mainData);
                const { open, high, low, close } = main;
                    html += `<div class="legend-row">
                        <span class="legend-label">O</span> <span class="legend-value">${open.toFixed(2)}</span>
                        <span class="legend-label">H</span> <span class="legend-value">${high.toFixed(2)}</span>
                        <span class="legend-label">L</span> <span class="legend-value">${low.toFixed(2)}</span>
                        <span class="legend-label">C</span> <span class="legend-value">${close.toFixed(2)}</span>
                    </div>`;
                } else {
                    // Line chart main series
                const main = /** @type {any} */ (mainData);
                const val = main.value !== undefined ? main.value : main.close;
                    html += `<div class="legend-row">
                        <span class="legend-label">Value</span> <span class="legend-value">${val.toFixed(2)}</span>
                    </div>`;
                }
            }

            // Other Series
            param.seriesData.forEach((value, series) => {
                if (series === mainSeries) return;
                
                let name = '';
                let color = '#d1d4dc';
                
                // Find name/color from maps
                for (const [n, s] of lineSeriesMap.entries()) {
                    if (s === series) {
                        name = n;
                        color = s.options().color;
                        break;
                    }
                }
                if (!name) {
                    for (const [n, s] of histSeriesMap.entries()) {
                        if (s === series) {
                            name = n;
                            color = s.options().color;
                            break;
                        }
                    }
                }

                if (name) {
                  const v = /** @type {any} */ (value);
                  const val = v.value !== undefined ? v.value : v.close;
                    html += `<div class="legend-row" style="color: ${color}">
                        <span>${name}</span>: <span>${val.toFixed(2)}</span>
                    </div>`;
                }
            });

            legendContainer.innerHTML = html;
        });

        const ro = new ResizeObserver(entries => {
          if (!entries.length) return;
          const { width, height: h } = entries[0].contentRect;
          chart.applyOptions({ width, height: h });
        });
        ro.observe(container);
        resizeObserver = ro;
    }
  }

  let resizeObserver;

  function applyData(data) {
    if (!chart || !mainSeries || !data) return;

    // Main Series Data
    // If chartType is line, we expect data.main or data.candles (if formatted as line data)
    // But usually data.candles is OHLC.
    // Let's assume if chartType is line, we look for data.main, OR we use the first line in data.lines as main?
    // Or we can just reuse data.candles if the user passes line data there?
    // Let's stick to: if chartType='line', we expect data.main.
    
    if (chartType === 'candlestick') {
        if (Array.isArray(data.candles)) {
            mainSeries.setData(data.candles);
        }
    } else {
        if (Array.isArray(data.main)) {
            mainSeries.setData(data.main);
        }
    }

    // Markers on main series
    if (Array.isArray(data.markers)) {
      if (!mainSeriesMarkers) {
        mainSeriesMarkers = createSeriesMarkers(mainSeries, data.markers);
      } else {
        // createSeriesMarkers returns an object with setMarkers? 
        // The plugin returns an API to manipulate markers?
        // Actually createSeriesMarkers(series, markers) creates them. 
        // To update, we might need to detach and recreate or use setMarkers if available.
        // The plugin documentation says: createSeriesMarkers(series, markers) -> { detach: () => void, setMarkers: (markers) => void }
        if (mainSeriesMarkers.setMarkers) {
             mainSeriesMarkers.setMarkers(data.markers);
        } else {
             // Fallback if setMarkers not available (older version?)
             mainSeriesMarkers.detach();
             mainSeriesMarkers = createSeriesMarkers(mainSeries, data.markers);
        }
      }
    }

    // Lines
    // Note: removing and re-adding series on toggle can occasionally glitch in lightweight-charts.
    // Keep series instances and just clear/set data so re-toggle is reliable.
    const incomingLineNames = new Set();
    if (Array.isArray(data.lines)) {
      data.lines.forEach((line, index) => {
        const name = line.name || `line-${index}`;
        incomingLineNames.add(name);
        let series = lineSeriesMap.get(name);
        if (!series) {
          series = chart.addSeries(LineSeries, {
            color: line.color || '#2962FF',
            lineWidth: line.lineWidth || 2,
            lineStyle: line.lineStyle || 0,
            priceScaleId: line.priceScaleId || 'right'
          });
          lineSeriesMap.set(name, series);
        } else {
          series.applyOptions({
            color: line.color || '#2962FF',
            lineWidth: line.lineWidth || 2,
            lineStyle: line.lineStyle || 0,
            priceScaleId: line.priceScaleId || 'right'
          });
        }
        series.setData(Array.isArray(line.data) ? line.data : []);
      });
    }
    // Clear stale line series
    for (const [name, series] of lineSeriesMap.entries()) {
      if (!incomingLineNames.has(name)) {
        series.setData([]);
      }
    }

    // Histograms
    const incomingHistNames = new Set();
    if (Array.isArray(data.histograms)) {
      data.histograms.forEach((hist, index) => {
        const name = hist.name || `hist-${index}`;
        incomingHistNames.add(name);
        let series = histSeriesMap.get(name);
        if (!series) {
          series = chart.addSeries(HistogramSeries, {
            color: hist.color || '#26a69a',
            priceScaleId: hist.priceScaleId || 'right',
            base: hist.base || 0
          });
          histSeriesMap.set(name, series);
        } else {
          series.applyOptions({
            color: hist.color || '#26a69a',
            priceScaleId: hist.priceScaleId || 'right',
            base: hist.base || 0
          });
        }
        series.setData(Array.isArray(hist.data) ? hist.data : []);
      });
    }
    // Clear stale hist series
    for (const [name, series] of histSeriesMap.entries()) {
      if (!incomingHistNames.has(name)) {
        series.setData([]);
      }
    }

    // Only fit content if this is the first load or explicitly requested via key change
    if (!hasFitContent) {
        chart.timeScale().fitContent();
        hasFitContent = true;
    }
  }

  onMount(() => {
    initChart();
  });

  $: if (chart) {
    applyData(tvlcData);
  }

  onDestroy(() => {
    if (resizeObserver && container) resizeObserver.unobserve(container);
    lineSeriesMap.forEach(series => chart.removeSeries(series));
    histSeriesMap.forEach(series => chart.removeSeries(series));
    if (mainSeriesMarkers) mainSeriesMarkers.detach();
    if (mainSeries) chart.removeSeries(mainSeries);
    if (chart) chart.remove();
  });
</script>

<div class="chart-wrapper" style={`width: 100%; height: ${height}px; position: relative;`}>
    <div bind:this={legendContainer} class="legend"></div>
    <div bind:this={container} style="width: 100%; height: 100%;"></div>
</div>

<style>
    .legend {
        position: absolute;
        top: 12px;
        left: 12px;
        z-index: 2;
        font-size: 12px;
        font-family: 'Monaco', 'Consolas', monospace;
        line-height: 18px;
        font-weight: 300;
        pointer-events: none;
        color: #d1d4dc;
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    :global(.legend-row) {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    :global(.legend-label) {
        color: #787b86;
        margin-right: 2px;
    }
    :global(.legend-value) {
        margin-right: 8px;
    }
</style>
