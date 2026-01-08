<script>
  import { onMount, onDestroy } from 'svelte';
  import { createChart, CandlestickSeries, LineSeries, HistogramSeries, createSeriesMarkers } from 'lightweight-charts';

  export let tvlcData = { candles: [], lines: [], markers: [], histograms: [] };
  export let height = 520;
  export let options = {};
  export let chartType = 'candlestick'; // 'candlestick' or 'line'
  export let mainSeriesOptions = {};

  let container;
  let legendContainer;
  let chart;
  let mainSeries; // Renamed from candleSeries
  let mainSeriesMarkers; // Renamed from candleSeriesMarkers
  const lineSeriesMap = new Map();
  const histSeriesMap = new Map();

  const baseOptions = {
    layout: { background: { color: '#131722' }, textColor: '#d1d4dc' },
    grid: { vertLines: { color: '#2B2B43' }, horzLines: { color: '#2B2B43' } },
    timeScale: { timeVisible: true, secondsVisible: false },
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
                    const { open, high, low, close } = mainData;
                    html += `<div class="legend-row">
                        <span class="legend-label">O</span> <span class="legend-value">${open.toFixed(2)}</span>
                        <span class="legend-label">H</span> <span class="legend-value">${high.toFixed(2)}</span>
                        <span class="legend-label">L</span> <span class="legend-value">${low.toFixed(2)}</span>
                        <span class="legend-label">C</span> <span class="legend-value">${close.toFixed(2)}</span>
                    </div>`;
                } else {
                    // Line chart main series
                    const val = mainData.value !== undefined ? mainData.value : mainData.close;
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
                    const val = value.value !== undefined ? value.value : value.close;
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
    const incomingLineNames = new Set();
    if (Array.isArray(data.lines)) {
      data.lines.forEach(line => {
        const name = line.name || `line-${Math.random()}`;
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
        if (Array.isArray(line.data)) {
          series.setData(line.data);
        }
      });
    }
    // Remove stale line series
    for (const [name, series] of lineSeriesMap.entries()) {
      if (!incomingLineNames.has(name)) {
        chart.removeSeries(series);
        lineSeriesMap.delete(name);
      }
    }

    // Histograms
    const incomingHistNames = new Set();
    if (Array.isArray(data.histograms)) {
      data.histograms.forEach(hist => {
        const name = hist.name || `hist-${Math.random()}`;
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
        if (Array.isArray(hist.data)) {
          series.setData(hist.data);
        }
      });
    }
    // Remove stale hist series
    for (const [name, series] of histSeriesMap.entries()) {
      if (!incomingHistNames.has(name)) {
        chart.removeSeries(series);
        histSeriesMap.delete(name);
      }
    }

    // Only fit content if this is the first load or explicitly requested
    // For now, we'll assume if we have data and it's the first time applying it (or very different), we fit.
    // But to avoid resetting zoom on refresh, we can check if the time scale is already set.
    // A simple heuristic: if the visible range is not set (default), fit content.
    // However, TVLC doesn't expose "is zoomed" easily without checking visibleLogicalRange.
    // For this specific request, let's just fit content if it's the initial data load.
    
    // We can use a flag or check if the chart has been interacted with.
    // For simplicity, let's just fit content. If the user wants to preserve zoom, we'd need more complex logic.
    // Actually, let's NOT fit content on every update if we can avoid it.
    // But since we are replacing data (setData), TVLC might reset anyway.
    // Let's stick to fitContent for now as it ensures the new data is visible.
    chart.timeScale().fitContent();
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
