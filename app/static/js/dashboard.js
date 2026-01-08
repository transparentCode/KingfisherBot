class TradingDashboard {
    constructor() {
        this.currentSymbol = 'BTCUSDT';
        this.currentTimeframe = '1h';
        this.activeIndicators = new Map();
        this.indicatorTraces = new Map();
        this.availableIndicators = null;
        this.chartData = null;
        this.isLiveMode = false;
        this.drawMode = null;
        this.drawPoints = [];
        this.zoomState = null;
        this.searchTimeout = null;
        this.lastPrice = null;
        this.assets = new Map(); // Store asset status

        // Performance optimizations
        this.isLoading = false; // Prevent concurrent loads
        this.updateTimeout = null; // Debounce updates

        this.init();
    }

    /**
     * Initialize the dashboard
     */
    async init() {
        try {
            await this.loadAvailableIndicators();
            this.startAssetPolling(); // Start polling for assets
            // await this.loadChart(); // Wait for assets to load first
            this.setupEventListeners();
            this.updateActiveIndicatorsList();

            console.log('✅ Trading Dashboard initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize dashboard:', error);
            this.showError('Failed to initialize dashboard');
        }
    }

    debugIndicatorData(indicatorData) {
        console.log('🔍 Debugging indicator data:', {
            indicator_name: indicatorData.indicator_name,
            plot_data: indicatorData.plot_data,
            data_length: indicatorData.plot_data?.data?.length,
            first_trace: indicatorData.plot_data?.data?.[0],
            sample_data: indicatorData.plot_data?.data?.[0]?.y?.slice(0, 5)
        });
    }

    /**
     * Optimized event listeners setup
     */
    setupEventListeners() {
        // Symbol change
        $('#active-symbol').on('change', (e) => {
            this.currentSymbol = e.target.value;
            this.updateStatusBadge();
            this.debouncedLoadChart();
        });

        // Timeframe change
        $('input[name="timeframe"]').on('change', (e) => {
            this.currentTimeframe = e.target.value;
            this.debouncedLoadChart();
        });

        // Chart controls
        $('#refresh-chart').on('click', () => this.loadChart());
        $('#add-indicator-btn').on('click', () => this.showAddIndicatorModal());
        $('#start-live-data').on('click', () => this.startLiveData());
        $('#pause-live-data').on('click', () => this.pauseLiveData());

        // Search with debouncing
        $('#symbol-search').on('input', (e) => {
            clearTimeout(this.searchTimeout);
            this.searchTimeout = setTimeout(() => {
                this.searchSymbols(e.target.value);
            }, 300);
        });
    }

    /**
     * Poll for asset status
     */
    async startAssetPolling() {
        const poll = async () => {
            try {
                const response = await fetch('/api/system/assets');
                const data = await response.json();
                if (data.success) {
                    this.updateAssetList(data.assets);
                }
            } catch (error) {
                console.error('Failed to fetch assets:', error);
            }
            setTimeout(poll, 5000); // Poll every 5 seconds
        };
        poll();
    }

    /**
     * Update asset list and status
     */
    updateAssetList(assets) {
        const select = $('#active-symbol');
        const currentVal = select.val();
        
        // Update internal state
        const oldStatus = this.assets.get(this.currentSymbol);
        this.assets.clear();
        assets.forEach(a => this.assets.set(a.symbol, a.status));

        // Rebuild options if list changed (simple check: length or first item)
        // For robustness, we'll just rebuild if the count is different or if we are in initial load
        // A better diffing could be done but this is likely fine for now.
        // Actually, let's just rebuild the options to ensure status icons are up to date in the dropdown too
        
        const currentOptions = select.find('option').map((_, opt) => opt.value).get();
        const newSymbols = assets.map(a => a.symbol);
        
        // Check if we need to rebuild dropdown (symbols changed)
        const symbolsChanged = JSON.stringify(currentOptions) !== JSON.stringify(newSymbols);
        
        if (symbolsChanged || select.find('option').first().text() === 'Loading assets...') {
            select.empty();
            assets.forEach(asset => {
                // We won't put the icon in the option text as it might look messy, 
                // but we could. Let's stick to just the symbol for now in the dropdown
                // and rely on the badge for status.
                const option = new Option(asset.symbol, asset.symbol);
                select.append(option);
            });

            // Restore selection
            if (currentVal && newSymbols.includes(currentVal)) {
                select.val(currentVal);
            } else if (assets.length > 0) {
                // Default to first if current selection is invalid/empty
                select.val(assets[0].symbol);
                this.currentSymbol = assets[0].symbol;
                this.loadChart();
            }
        }

        // Update status badge
        this.updateStatusBadge();
        
        // If status of current symbol changed, maybe reload chart?
        // For now, just update the badge.
    }

    updateStatusBadge() {
        const status = this.assets.get(this.currentSymbol);
        const badge = $('#asset-status-badge');
        
        badge.removeClass('bg-secondary bg-success bg-warning bg-danger text-dark');
        badge.empty();

        if (!status) {
            badge.addClass('bg-secondary').html('<i class="fas fa-question"></i>');
            return;
        }

        switch (status) {
            case 'READY':
                badge.addClass('bg-success').html('READY');
                break;
            case 'FETCHING':
                badge.addClass('bg-warning text-dark').html('<i class="fas fa-sync fa-spin me-1"></i>SYNC');
                break;
            case 'PENDING':
                badge.addClass('bg-secondary').html('PENDING');
                break;
            default:
                badge.addClass('bg-danger').html(status);
        }
    }

    /**
     * Debounced chart loading
     */
    debouncedLoadChart() {
        clearTimeout(this.updateTimeout);
        this.updateTimeout = setTimeout(() => {
            this.loadChart();
        }, 500); // 500ms debounce
    }

    /**
     * Load available indicators from API
     */
    async loadAvailableIndicators() {
        try {
            const response = await fetch('/api/indicators/available');
            const data = await response.json();

            if (data.success) {
                this.availableIndicators = data.categories;
                console.log('📊 Loaded indicators:', Object.keys(this.availableIndicators));
            } else {
                throw new Error(data.error || 'Failed to load indicators');
            }
        } catch (error) {
            console.error('❌ Error loading indicators:', error);
            throw error;
        }
    }

    /**
     * Setup all event handlers
     */
    setupEventHandlers() {
        // Symbol selection
        $('#active-symbol').on('change', (e) => {
            this.currentSymbol = e.target.value;
            this.updateChartTitle();
            this.loadChart();
        });

        // Timeframe selection
        $('input[name="timeframe"]').on('change', (e) => {
            this.currentTimeframe = e.target.value;
            this.updateChartTitle();
            this.loadChart();
        });

        // Lookback days
        $('#lookback-days').on('change', (e) => {
            this.lookbackDays = parseInt(e.target.value);
            if (!this.isLiveMode) {
                this.loadChart();
            }
        });

        // Live data controls
        $('#start-live-data').on('click', () => this.startLiveData());
        $('#pause-live-data').on('click', () => this.pauseLiveData());

        // Chart controls
        $('#refresh-chart').on('click', () => this.loadChart());
        $('#fullscreen-chart').on('click', () => this.toggleFullscreen());

        // Indicator management
        $('#add-indicator-btn').on('click', () => this.showAddIndicatorModal());

        // Symbol search
        $('#symbol-search').on('input', (e) => this.searchSymbols(e.target.value));
    }

    /**
     * Update chart title
     */
    updateChartTitle() {
        $('#chart-title').text(`${this.currentSymbol} - ${this.currentTimeframe}`);
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(status) {
        const statusElement = $('#connection-status .status-indicator');
        const statusText = $('#connection-status');

        statusElement.removeClass('status-connected status-disconnected status-connecting');

        switch(status) {
            case 'connected':
                statusElement.addClass('status-connected');
                statusText.html('<span class="status-indicator status-connected"></span>Live');
                break;
            case 'disconnected':
                statusElement.addClass('status-disconnected');
                statusText.html('<span class="status-indicator status-disconnected"></span>Offline');
                break;
            case 'connecting':
                statusElement.addClass('status-connecting');
                statusText.html('<span class="status-indicator status-connecting"></span>Connecting');
                break;
        }
    }

    /**
     * Load main chart with base candlestick data
     */
    async loadChart() {
        if (this.isLoading) {
            console.log('⏳ Chart loading already in progress, skipping...');
            return;
        }

        this.isLoading = true;
        this.showChartLoading(true);

        try {
            // Load base chart first
            await this.loadBaseCandlestickChart();

            // Then load indicators if any are active
            if (this.activeIndicators.size > 0) {
                await this.loadAllActiveIndicators();
            }

            this.updateChartTitle();

        } catch (error) {
            console.error('❌ Error loading chart:', error);
            this.showError('Failed to load chart data');
        } finally {
            this.isLoading = false;
            this.showChartLoading(false);
        }
    }

    /**
     * Clear all indicator traces from chart
     */
    clearAllIndicatorTraces() {
        try {
            const plotlyDiv = document.getElementById('trading-chart');

            if (!plotlyDiv.data || plotlyDiv.data.length === 0) return;

            // Keep only candlestick and volume traces (first 2 traces)
            const baseCandlestickTrace = plotlyDiv.data[0];
            const baseVolumeTrace = plotlyDiv.data[1];

            // Clear all traces and re-add base traces
            Plotly.deleteTraces(plotlyDiv, Array.from({length: plotlyDiv.data.length}, (_, i) => i));
            Plotly.addTraces(plotlyDiv, [baseCandlestickTrace, baseVolumeTrace]);

            // Reset indicator traces tracking
            this.indicatorTraces = new Map();

            console.log('✅ Cleared all indicator traces');
        } catch (error) {
            console.error('❌ Error clearing indicator traces:', error);
        }
    }


    /**
     * Load all active indicators at once
     */
    async loadAllActiveIndicators() {
        if (this.activeIndicators.size === 0) return;

        try {
            // Clear existing indicator traces first
            this.clearIndicatorTraces();

            // Fetch all indicators in parallel
            const indicatorPromises = Array.from(this.activeIndicators.entries())
                .filter(([_, config]) => config.visible)
                .map(([indicatorId, config]) =>
                    this.fetchIndicatorData(indicatorId, config.params)
                        .catch(error => {
                            console.error(`Failed to load ${config.name}:`, error);
                            return null; // Don't fail entire batch
                        })
                );

            const indicatorResults = await Promise.all(indicatorPromises);

            // Add all valid indicators to chart
            const validResults = indicatorResults.filter(result => result !== null);
            if (validResults.length > 0) {
                this.addAllIndicatorsToChart(validResults);
            }

        } catch (error) {
            console.error('❌ Error loading indicators:', error);
        }
    }

    /**
     * Clear only indicator traces, keep candlestick and volume
     */
    clearIndicatorTraces() {
        try {
            const plotlyDiv = document.getElementById('trading-chart');
            if (!plotlyDiv.data || plotlyDiv.data.length <= 2) return;

            // Keep first 2 traces (candlestick + volume), remove the rest
            const tracesToRemove = [];
            for (let i = plotlyDiv.data.length - 1; i >= 2; i--) {
                tracesToRemove.push(i);
            }

            if (tracesToRemove.length > 0) {
                Plotly.deleteTraces(plotlyDiv, tracesToRemove);
            }

            // Reset indicator tracking
            this.indicatorTraces.clear();

        } catch (error) {
            console.error('❌ Error clearing indicator traces:', error);
        }
    }

    /**
     * Add all indicators to chart at once
     */
    addAllIndicatorsToChart(indicatorResults) {
        try {
            const plotlyDiv = document.getElementById('trading-chart');

            if (!plotlyDiv.data) {
                console.warn('Chart not initialized yet');
                return;
            }

            // Separate indicators by type (overlay vs subplot)
            const overlayTraces = [];
            const subplotTraces = [];
            const layoutUpdates = {};
            let subplotCount = 0;

            indicatorResults.forEach(indicatorData => {
                if (!indicatorData || !indicatorData.plot_data) {
                    console.warn('Invalid indicator data:', indicatorData);
                    return;
                }

                const plotData = indicatorData.plot_data;
                console.log(`Processing indicator: ${indicatorData.indicator_name}`, plotData);

                if (plotData && plotData.data && Array.isArray(plotData.data)) {
                    plotData.data.forEach(trace => {
                        // Ensure timestamps are properly formatted
                        if (trace.x && Array.isArray(trace.x)) {
                            trace.x = trace.x.map(timestamp =>
                                timestamp instanceof Date ? timestamp : new Date(timestamp)
                            );
                        }

                        // Apply proper styling
                        if (trace.line && !trace.line.width) {
                            trace.line.width = 2;
                        }

                        // Determine if this is a subplot indicator
                        const isSubplotIndicator = this.isSubplotIndicator(indicatorData.indicator_name);

                        if (isSubplotIndicator) {
                            subplotCount++;
                            const yAxisName = `y${2 + subplotCount}`; // y3, y4, etc.
                            trace.yaxis = yAxisName;
                            subplotTraces.push(trace);

                            // Add corresponding y-axis configuration
                            layoutUpdates[`yaxis${2 + subplotCount}`] = this.getSubplotAxisConfig(subplotCount, indicatorData.indicator_name);
                        } else {
                            // Overlay on main chart
                            trace.yaxis = 'y';
                            overlayTraces.push(trace);
                        }
                    });
                }
            });

            // Add all traces at once
            const allTraces = [...overlayTraces, ...subplotTraces];
            if (allTraces.length > 0) {
                Plotly.addTraces(plotlyDiv, allTraces);
                console.log(`✅ Added ${allTraces.length} indicator traces to chart`);
            }

            // Update layout for subplots
            if (subplotCount > 0) {
                this.updateLayoutForSubplots(plotlyDiv, subplotCount, layoutUpdates);
            }

        } catch (error) {
            console.error('❌ Error adding indicators to chart:', error);
        }
    }

    /**
     * Update chart layout to accommodate subplots
     */
    updateLayoutForSubplots(plotlyDiv, subplotCount, layoutUpdates) {
        try {
            const currentLayout = plotlyDiv.layout;

            // Calculate main chart domain - leave more space for subplots
            const volumeHeight = 0.15;
            const subplotTotalHeight = subplotCount * 0.17; // 15% + 2% spacing per subplot
            const mainChartBottom = 0.02 + volumeHeight + subplotTotalHeight + 0.05; // Add padding

            const updatedLayout = {
                ...currentLayout,
                ...layoutUpdates,
                // Adjust main chart domain
                yaxis: {
                    ...currentLayout.yaxis,
                    domain: [mainChartBottom, 0.98] // Leave 2% at top
                },
                // Volume subplot - fixed position
                yaxis2: {
                    ...currentLayout.yaxis2,
                    domain: [0.02, 0.17], // Fixed volume position
                    title: 'Volume',
                    gridcolor: '#404040',
                    color: '#ffffff',
                    side: 'right',
                    showticklabels: false
                }
            };

            Plotly.relayout(plotlyDiv, updatedLayout);
            console.log('✅ Updated layout for subplots', updatedLayout);
        } catch (error) {
            console.error('❌ Error updating layout for subplots:', error);
        }
    }


    /**
     * Check if indicator should be displayed in subplot
     */
    isSubplotIndicator(indicatorName) {
        const subplotIndicators = ['RSI', 'MACD', 'Stochastic', 'Williams %R', 'CCI', 'ROC', 'RegimeMetrics'];
        return subplotIndicators.some(name =>
            indicatorName.toUpperCase().includes(name.toUpperCase())
        );
    }

    /**
     * Get subplot axis configuration
     */
    getSubplotAxisConfig(subplotIndex, indicatorName) {
        const baseConfig = {
            title: indicatorName,
            gridcolor: '#404040',
            color: '#ffffff',
            side: 'right',
            showticklabels: true,
            domain: this.getSubplotDomain(subplotIndex)
        };

        // RSI specific configuration
        if (indicatorName.toUpperCase().includes('RSI')) {
            return {
                ...baseConfig,
                range: [0, 100],
                tickmode: 'linear',
                dtick: 20, // Show ticks at 0, 20, 40, 60, 80, 100
                fixedrange: false,
                // Add reference lines for RSI
                shapes: [
                    {
                        type: 'line',
                        x0: 0, x1: 1, xref: 'paper',
                        y0: 70, y1: 70,
                        line: { color: '#ff4757', width: 1, dash: 'dash' }
                    },
                    {
                        type: 'line',
                        x0: 0, x1: 1, xref: 'paper',
                        y0: 30, y1: 30,
                        line: { color: '#00ff88', width: 1, dash: 'dash' }
                    }
                ]
            };
        }

        return baseConfig;
    }

    /**
     * Calculate subplot domain based on number of subplots
     */
    getSubplotDomain(subplotIndex) {
        const subplotHeight = 0.15; // Each subplot takes 15% of chart height
        const spacing = 0.02; // 2% spacing between subplots
        const volumeHeight = 0.15; // Volume subplot height
        const volumeBottom = 0.02; // Volume starts at 2% from bottom

        // Calculate subplot position from bottom up
        const subplotBottom = volumeBottom + volumeHeight + spacing + ((subplotIndex - 1) * (subplotHeight + spacing));
        const subplotTop = subplotBottom + subplotHeight;

        return [subplotBottom, subplotTop];
    }


    /**
     * Fetch indicator data from API
     */
    async fetchIndicatorData(indicatorId, params) {
        const paramString = Object.entries(params)
            .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
            .join('&');

        const url = `/api/indicator/${indicatorId}/calculate?symbol=${this.currentSymbol}&interval=${this.currentTimeframe}&limit=500&${paramString}`;

        const response = await fetch(url);
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Failed to calculate indicator');
        }

        return data;
    }

    /**
     * Load base candlestick chart
     */
    async loadBaseCandlestickChart() {
        try {
            const url = `/api/candle-data/${this.currentSymbol}?timeframe=${this.currentTimeframe}&limit=500`;
            const response = await fetch(url);
            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Failed to load market data');
            }

            this.chartData = data.data;

            // Convert candle data to arrays for Plotly
            const timestamps = data.data.candles.map(candle => new Date(candle.time * 1000));
            const opens = data.data.candles.map(candle => candle.open);
            const highs = data.data.candles.map(candle => candle.high);
            const lows = data.data.candles.map(candle => candle.low);
            const closes = data.data.candles.map(candle => candle.close);
            const volumes = data.data.candles.map(candle => candle.volume);

            // Enhanced candlestick trace
            const candlestickTrace = {
                x: timestamps,
                open: opens,
                high: highs,
                low: lows,
                close: closes,
                type: 'candlestick',
                name: this.currentSymbol,
                increasing: {
                    line: { color: '#00ff88', width: 1 },
                    fillcolor: '#00ff88'
                },
                decreasing: {
                    line: { color: '#ff4757', width: 1 },
                    fillcolor: '#ff4757'
                },
                hoverlabel: {
                    bgcolor: 'rgba(0,0,0,0.8)',
                    bordercolor: '#fff',
                    font: { color: '#fff' }
                }
            };

            // Volume trace (subplot)
            const volumeTrace = {
                x: timestamps,
                y: volumes,
                type: 'bar',
                name: 'Volume',
                marker: {
                    color: volumes.map((vol, i) => closes[i] >= opens[i] ? '#00ff8844' : '#ff475744')
                },
                yaxis: 'y2',
                showlegend: false
            };

            // Base layout - will be updated as indicators are added
            const layout = {
                title: {
                    text: `${this.currentSymbol} - ${this.currentTimeframe}`,
                    font: { color: '#ffffff', size: 16 }
                },
                xaxis: {
                    type: 'date',
                    gridcolor: '#404040',
                    color: '#ffffff',
                    rangeslider: { visible: false },
                    showspikes: true,
                    spikecolor: '#ffffff',
                    spikethickness: 1,
                    spikedash: 'dot',
                    spikemode: 'across',
                    showticklabels: true,
                    tickformat: '%H:%M\n%d/%m'
                },
                yaxis: {
                    title: 'Price (USDT)',
                    gridcolor: '#404040',
                    color: '#ffffff',
                    side: 'right',
                    showspikes: true,
                    spikecolor: '#ffffff',
                    spikethickness: 1,
                    spikedash: 'dot',
                    spikemode: 'across',
                    domain: [0.4, 1] // Will be adjusted for subplots
                },
                yaxis2: {
                    title: 'Volume',
                    gridcolor: '#404040',
                    color: '#ffffff',
                    side: 'right',
                    domain: [0.25, 0.35],
                    showticklabels: false
                },
                paper_bgcolor: '#1a1a1a',
                plot_bgcolor: '#1a1a1a',
                font: { color: '#ffffff' },
                showlegend: true,
                legend: {
                    x: 0,
                    y: 1,
                    bgcolor: 'rgba(0,0,0,0.5)',
                    bordercolor: '#404040'
                },
                margin: { t: 50, b: 50, l: 20, r: 80 },
                hovermode: 'x unified',
                dragmode: 'pan'
            };

            // Enhanced configuration
            const config = {
                responsive: true,
                displayModeBar: true,
                displaylogo: false,
                modeBarButtonsToRemove: [
                    'select2d', 'lasso2d', 'autoScale2d', 'hoverClosestCartesian',
                    'hoverCompareCartesian', 'toggleSpikelines'
                ],
                scrollZoom: true,
                doubleClick: 'reset+autosize'
            };

            // Create the chart with base traces only
            Plotly.newPlot('trading-chart', [candlestickTrace, volumeTrace], layout, config);

            // Add custom event listeners
            this.addChartEventListeners();

            // Update price ticker
            this.updatePriceTicker({
                close: closes,
                high: highs,
                low: lows,
                volume: volumes
            });

            console.log('✅ Base candlestick chart created');

        } catch (error) {
            console.error('❌ Error loading candlestick chart:', error);
            throw error;
        }
    }


    /**
     * Toggle drawing mode
     */
    toggleDrawMode(mode) {
        this.drawMode = this.drawMode === mode ? null : mode;
        this.drawPoints = [];

        const chartDiv = document.getElementById('trading-chart');
        chartDiv.style.cursor = this.drawMode ? 'crosshair' : 'default';

        // Update UI to show current mode
        this.updateDrawModeUI();
    }

    /**
     * Update draw mode UI
     */
    updateDrawModeUI() {
        // Remove active state from all draw buttons
        $('#draw-line-btn, #draw-rect-btn').removeClass('active');

        // Add active state to current mode
        if (this.drawMode === 'line') {
            $('#draw-line-btn').addClass('active');
        } else if (this.drawMode === 'rect') {
            $('#draw-rect-btn').addClass('active');
        }
    }

    /**
     * Add custom event listeners to chart
     */
    addChartEventListeners() {
        const chartDiv = document.getElementById('trading-chart');

        // Crosshair functionality
        chartDiv.on('plotly_hover', (data) => {
            this.showCrosshair(data);
            this.updateHoverInfo(data);
        });

        chartDiv.on('plotly_unhover', () => {
            this.hideCrosshair();
        });

        // Click events for drawing tools
        chartDiv.on('plotly_click', (data) => {
            if (this.drawMode) {
                this.handleDrawClick(data);
            }
        });

        // Zoom and pan events
        chartDiv.on('plotly_relayout', (eventData) => {
            this.handleZoomPan(eventData);
        });

        // Add keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            this.handleKeyboardShortcuts(e);
        });
    }

    /**
     * Handle zoom and pan events
     */
    handleZoomPan(eventData) {
        // Handle zoom/pan state changes
        if (eventData['xaxis.range[0]'] || eventData['xaxis.range[1]']) {
            console.log('Chart zoom/pan detected');

            // Store current zoom state
            this.zoomState = {
                xRange: [eventData['xaxis.range[0]'], eventData['xaxis.range[1]']],
                yRange: [eventData['yaxis.range[0]'], eventData['yaxis.range[1]']]
            };
        }

        // Handle autorange
        if (eventData['xaxis.autorange'] || eventData['yaxis.autorange']) {
            this.zoomState = null;
        }
    }

    /**
     * Handle drawing clicks
     */
    handleDrawClick(data) {
        const point = data.points[0];
        if (!point) return;

        this.drawPoints.push({
            x: point.x,
            y: point.y
        });

        if (this.drawMode === 'line' && this.drawPoints.length === 2) {
            this.drawLine(this.drawPoints[0], this.drawPoints[1]);
            this.drawPoints = [];
        } else if (this.drawMode === 'rect' && this.drawPoints.length === 2) {
            this.drawRectangle(this.drawPoints[0], this.drawPoints[1]);
            this.drawPoints = [];
        }
    }

    /**
     * Draw rectangle on chart
     */
    drawRectangle(point1, point2) {
        const chartDiv = document.getElementById('trading-chart');
        const currentShapes = chartDiv.layout.shapes || [];

        const newShape = {
            type: 'rect',
            x0: point1.x,
            y0: point1.y,
            x1: point2.x,
            y1: point2.y,
            line: {
                color: '#00ff88',
                width: 2
            },
            fillcolor: 'rgba(0, 255, 136, 0.1)'
        };

        Plotly.relayout(chartDiv, {
            shapes: [...currentShapes, newShape]
        });
    }

    /**
     * Draw line on chart
     */
    drawLine(point1, point2) {
        const chartDiv = document.getElementById('trading-chart');
        const currentShapes = chartDiv.layout.shapes || [];

        const newShape = {
            type: 'line',
            x0: point1.x,
            y0: point1.y,
            x1: point2.x,
            y1: point2.y,
            line: {
                color: '#00ff88',
                width: 2
            }
        };

        Plotly.relayout(chartDiv, {
            shapes: [...currentShapes, newShape]
        });
    }

    /**
     * Show crosshair on hover
     */
    showCrosshair(data) {
        const chartDiv = document.getElementById('trading-chart');
        const point = data.points[0];

        if (!point) return;

        // Update crosshair lines
        const update = {
            shapes: [
                // Vertical line
                {
                    type: 'line',
                    x0: point.x,
                    x1: point.x,
                    y0: 0,
                    y1: 1,
                    yref: 'paper',
                    line: {
                        color: '#ffffff',
                        width: 1,
                        dash: 'dot'
                    }
                },
                // Horizontal line
                {
                    type: 'line',
                    x0: 0,
                    x1: 1,
                    xref: 'paper',
                    y0: point.y,
                    y1: point.y,
                    line: {
                        color: '#ffffff',
                        width: 1,
                        dash: 'dot'
                    }
                }
            ]
        };

        Plotly.relayout(chartDiv, update);
    }

    /**
     * Zoom functions
     */
    zoomIn() {
        const chartDiv = document.getElementById('trading-chart');
        const xRange = chartDiv.layout.xaxis.range;
        const yRange = chartDiv.layout.yaxis.range;

        if (xRange && yRange) {
            const xCenter = (xRange[0] + xRange[1]) / 2;
            const yCenter = (yRange[0] + yRange[1]) / 2;
            const xWidth = (xRange[1] - xRange[0]) * 0.8;
            const yWidth = (yRange[1] - yRange[0]) * 0.8;

            Plotly.relayout(chartDiv, {
                'xaxis.range': [xCenter - xWidth/2, xCenter + xWidth/2],
                'yaxis.range': [yCenter - yWidth/2, yCenter + yWidth/2]
            });
        }
    }

    zoomOut() {
        const chartDiv = document.getElementById('trading-chart');
        const xRange = chartDiv.layout.xaxis.range;
        const yRange = chartDiv.layout.yaxis.range;

        if (xRange && yRange) {
            const xCenter = (xRange[0] + xRange[1]) / 2;
            const yCenter = (yRange[0] + yRange[1]) / 2;
            const xWidth = (xRange[1] - xRange[0]) * 1.2;
            const yWidth = (yRange[1] - yRange[0]) * 1.2;

            Plotly.relayout(chartDiv, {
                'xaxis.range': [xCenter - xWidth/2, xCenter + xWidth/2],
                'yaxis.range': [yCenter - yWidth/2, yCenter + yWidth/2]
            });
        }
    }

    /**
     * Handle keyboard shortcuts
     */
    handleKeyboardShortcuts(event) {
        const chartDiv = document.getElementById('trading-chart');

        switch(event.key) {
            case '+':
            case '=':
                this.zoomIn();
                break;
            case '-':
                this.zoomOut();
                break;
            case 'r':
            case 'R':
                Plotly.relayout(chartDiv, { 'xaxis.autorange': true, 'yaxis.autorange': true });
                break;
            case 'f':
            case 'F':
                this.toggleFullscreen();
                break;
            case 'Escape':
                this.exitDrawMode();
                break;
        }
    }

    /**
     * Exit draw mode
     */
    exitDrawMode() {
        this.drawMode = null;
        this.drawPoints = [];
        const chartDiv = document.getElementById('trading-chart');
        chartDiv.style.cursor = 'default';
        this.updateDrawModeUI();
    }


    /**
     * Update hover information panel
     */
    updateHoverInfo(data) {
        const point = data.points[0];
        if (!point || point.data.type !== 'candlestick') return;

        const hoverInfo = `
            <div class="hover-info bg-dark border rounded p-2 position-absolute"
                 style="top: 10px; left: 10px; z-index: 1000;">
                <div><strong>Time:</strong> ${new Date(point.x).toLocaleString()}</div>
                <div><strong>Open:</strong> $${point.open?.toFixed(2)}</div>
                <div><strong>High:</strong> $${point.high?.toFixed(2)}</div>
                <div><strong>Low:</strong> $${point.low?.toFixed(2)}</div>
                <div><strong>Close:</strong> $${point.close?.toFixed(2)}</div>
            </div>
        `;

        // Remove existing hover info
        $('.hover-info').remove();

        // Add new hover info
        $('#trading-chart').append(hoverInfo);
    }

    /**
     * Hide crosshair
     */
    hideCrosshair() {
        const chartDiv = document.getElementById('trading-chart');
        Plotly.relayout(chartDiv, { shapes: [] });
    }

    /**
     * Update price ticker with current data
     */
    updatePriceTicker(data) {
        if (!data || !data.close || data.close.length === 0) return;

        const currentPrice = data.close[data.close.length - 1];
        const previousPrice = data.close.length > 1 ? data.close[data.close.length - 2] : currentPrice;
        const priceChange = currentPrice - previousPrice;
        const priceChangePercent = ((priceChange / previousPrice) * 100);

        $('#current-price').text(`$${currentPrice.toLocaleString()}`);

        const changeElement = $('#price-change');
        const changeText = `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)} (${priceChangePercent.toFixed(2)}%)`;
        changeElement.text(changeText);
        changeElement.removeClass('positive negative');
        changeElement.addClass(priceChange >= 0 ? 'positive' : 'negative');

        // Update 24h stats
        const high = Math.max(...data.high);
        const low = Math.min(...data.low);
        const volume = data.volume ? data.volume.reduce((a, b) => a + b, 0) : 0;

        $('#price-high').text(`$${high.toLocaleString()}`);
        $('#price-low').text(`$${low.toLocaleString()}`);
        $('#price-volume').text(volume > 0 ? this.formatVolume(volume) : '--');
    }

    /**
     * Format volume for display
     */
    formatVolume(volume) {
        if (volume >= 1e9) return (volume / 1e9).toFixed(2) + 'B';
        if (volume >= 1e6) return (volume / 1e6).toFixed(2) + 'M';
        if (volume >= 1e3) return (volume / 1e3).toFixed(2) + 'K';
        return volume.toFixed(0);
    }

    /**
     * Show/hide chart loading overlay
     */
    showChartLoading(show) {
        const overlay = $('#chart-loading');
        if (show) {
            overlay.show();
        } else {
            overlay.hide();
        }
    }

    /**
     * Show add indicator modal
     */
    showAddIndicatorModal() {
        if (!this.availableIndicators) {
            this.showError('Indicators not loaded yet. Please wait...');
            return;
        }

        this.populateIndicatorModal();
        const modal = new bootstrap.Modal(document.getElementById('addIndicatorModal'));
        modal.show();
    }

    /**
     * Populate indicator modal with categories and indicators
     */
    populateIndicatorModal() {
        const categoriesContainer = $('#indicator-categories');
        const indicatorsContainer = $('#indicators-grid');

        // Clear containers
        categoriesContainer.empty();
        indicatorsContainer.empty();

        // Populate categories
        Object.keys(this.availableIndicators).forEach((category, index) => {
            const isActive = index === 0 ? 'active' : '';
            categoriesContainer.append(`
                <button type="button" class="list-group-item list-group-item-action ${isActive}"
                        data-category="${category}">
                    ${category}
                    <span class="badge bg-primary rounded-pill float-end">
                        ${this.availableIndicators[category].length}
                    </span>
                </button>
            `);
        });

        // Show first category indicators by default
        const firstCategory = Object.keys(this.availableIndicators)[0];
        this.showCategoryIndicators(firstCategory);

        // Category selection handler
        categoriesContainer.find('.list-group-item').on('click', (e) => {
            categoriesContainer.find('.list-group-item').removeClass('active');
            $(e.target).addClass('active');
            const category = $(e.target).data('category');
            this.showCategoryIndicators(category);
        });
    }

    /**
     * Show indicators for selected category
     */
    showCategoryIndicators(category) {
        const indicatorsContainer = $('#indicators-grid');
        indicatorsContainer.empty();

        const indicators = this.availableIndicators[category] || [];

        indicators.forEach(indicator => {
            const isAdded = this.activeIndicators.has(indicator.id);
            const buttonClass = isAdded ? 'btn-success' : 'btn-outline-primary';
            const buttonText = isAdded ? 'Added' : 'Add';
            const buttonIcon = isAdded ? 'fa-check' : 'fa-plus';

            indicatorsContainer.append(`
                <div class="indicator-card" data-indicator-id="${indicator.id}">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <h6 class="mb-0">${indicator.display_name}</h6>
                        <button class="btn ${buttonClass} btn-sm add-indicator-btn"
                                data-indicator-id="${indicator.id}" ${isAdded ? 'disabled' : ''}>
                            <i class="fas ${buttonIcon} me-1"></i>${buttonText}
                        </button>
                    </div>
                    <p class="text-muted small mb-0">${indicator.description}</p>
                </div>
            `);
        });

        // Add indicator button handlers
        indicatorsContainer.find('.add-indicator-btn').on('click', (e) => {
            e.stopPropagation();
            const indicatorId = $(e.target).closest('button').data('indicator-id');
            this.addIndicator(indicatorId);
        });
    }

    /**
     * Add indicator to chart
     */
    async addIndicator(indicatorId) {
        try {
            // Show settings modal first
            await this.showIndicatorSettings(indicatorId);
        } catch (error) {
            console.error('❌ Error adding indicator:', error);
            this.showError('Failed to add indicator');
        }
    }

    /**
     * Show indicator settings modal
     */
    async showIndicatorSettings(indicatorId, existingSettings = null) {
        try {
            // Get indicator schema
            const response = await fetch(`/api/indicators/${indicatorId}/schema`);
            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Failed to load indicator schema');
            }

            const indicator = data.indicator;

            // Setup modal
            $('#settings-modal-title').text(`${indicator.display_name} Settings`);

            // Generate form
            const formContainer = $('#indicator-settings-form');
            formContainer.empty();

            Object.entries(indicator.parameters).forEach(([paramName, paramConfig]) => {
                const currentValue = existingSettings?.[paramName] || paramConfig.default;
                formContainer.append(this.generateParameterInput(paramName, paramConfig, currentValue));
            });

            // Setup apply button
            $('#apply-indicator-settings').off('click').on('click', () => {
                this.applyIndicatorSettings(indicatorId, indicator.display_name);
            });

            // Show modal
            const modal = new bootstrap.Modal(document.getElementById('indicatorSettingsModal'));
            modal.show();

        } catch (error) {
            console.error('❌ Error loading indicator settings:', error);
            this.showError('Failed to load indicator settings');
        }
    }

    /**
     * Generate parameter input based on type
     */
    generateParameterInput(paramName, paramConfig, currentValue) {
        const formattedName = this.formatParameterName(paramName);

        let input = '';

        switch (paramConfig.type) {
            case 'int':
            case 'number':
                input = `
                    <input type="number" class="form-control"
                           id="param-${paramName}"
                           value="${currentValue}"
                           min="${paramConfig.min || ''}"
                           max="${paramConfig.max || ''}"
                           step="${paramConfig.type === 'int' ? '1' : '0.1'}">
                `;
                break;

            case 'select':
                const options = paramConfig.options.map(option =>
                    `<option value="${option}" ${option === currentValue ? 'selected' : ''}>${option}</option>`
                ).join('');
                input = `
                    <select class="form-select" id="param-${paramName}">
                        ${options}
                    </select>
                `;
                break;

            case 'boolean':
                input = `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox"
                               id="param-${paramName}" ${currentValue ? 'checked' : ''}>
                        <label class="form-check-label" for="param-${paramName}">
                            Enable
                        </label>
                    </div>
                `;
                break;

            default:
                input = `
                    <input type="text" class="form-control"
                           id="param-${paramName}"
                           value="${currentValue}">
                `;
        }

        return `
            <div class="mb-3">
                <label for="param-${paramName}" class="form-label">${formattedName}</label>
                ${input}
                <div class="form-text">${paramConfig.description || ''}</div>
            </div>
        `;
    }

    /**
     * Apply indicator settings and add to chart
     */
    async applyIndicatorSettings(indicatorId, displayName) {
        try {
            // Collect parameters
            const params = {};
            $('#indicator-settings-form input, #indicator-settings-form select').each(function() {
                const paramName = this.id.replace('param-', '');
                let value = $(this).val();

                if ($(this).attr('type') === 'checkbox') {
                    value = $(this).is(':checked');
                } else if ($(this).attr('type') === 'number') {
                    value = parseFloat(value);
                }
                params[paramName] = value;
            });

            // Add to active indicators
            this.activeIndicators.set(indicatorId, {
                id: indicatorId,
                name: displayName,
                params: params,
                visible: true
            });

            // Load just this indicator without reloading entire chart
            await this.loadSingleIndicator(indicatorId, params);

            // Update UI
            this.updateActiveIndicatorsList();

            // Close modals
            this.closeModals();

            console.log(`✅ Added indicator: ${displayName}`);

        } catch (error) {
            console.error('❌ Error applying indicator settings:', error);
            this.showError('Failed to add indicator');
        }
    }

    /**
     * Close all modals
     */
    closeModals() {
        ['indicatorSettingsModal', 'addIndicatorModal'].forEach(modalId => {
            const modalElement = document.getElementById(modalId);
            const modal = bootstrap.Modal.getInstance(modalElement);
            if (modal) modal.hide();
        });
    }

    /**
     * Load single indicator without full chart reload
     */
    async loadSingleIndicator(indicatorId, params) {
        try {
            const data = await this.fetchIndicatorData(indicatorId, params);
            this.addIndicatorToChart(data);
        } catch (error) {
            console.error(`❌ Error loading indicator ${indicatorId}:`, error);
            throw error;
        }
    }

    /**
     * Update active indicators list in sidebar
     */
    updateActiveIndicatorsList() {
        const container = $('#active-indicators-list');

        if (this.activeIndicators.size === 0) {
            container.html(`
                <div class="text-center text-muted py-3">
                    <i class="fas fa-info-circle mb-2"></i>
                    <p class="small mb-0">No indicators added yet</p>
                </div>
            `);
            return;
        }

        container.empty();

        this.activeIndicators.forEach((config, indicatorId) => {
            const visibilityIcon = config.visible ? 'fa-eye' : 'fa-eye-slash';
            const visibilityClass = config.visible ? 'text-success' : 'text-muted';

            container.append(`
                <div class="indicator-item" data-indicator-id="${indicatorId}">
                    <div class="d-flex justify-content-between align-items-center">
                        <div class="flex-grow-1">
                            <div class="fw-semibold">${config.name}</div>
                            <small class="text-muted">
                                ${Object.entries(config.params).slice(0, 2).map(([k, v]) => `${k}: ${v}`).join(', ')}
                            </small>
                        </div>
                        <div class="btn-group btn-group-sm" role="group">
                            <button class="btn btn-outline-light toggle-indicator"
                                    data-indicator-id="${indicatorId}"
                                    title="${config.visible ? 'Hide' : 'Show'} indicator">
                                <i class="fas ${visibilityIcon} ${visibilityClass}"></i>
                            </button>
                            <button class="btn btn-outline-light edit-indicator"
                                    data-indicator-id="${indicatorId}"
                                    title="Edit settings">
                                <i class="fas fa-cog"></i>
                            </button>
                            <button class="btn btn-outline-danger remove-indicator"
                                    data-indicator-id="${indicatorId}"
                                    title="Remove indicator">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            `);
        });

        // Add event handlers
        container.find('.toggle-indicator').on('click', (e) => {
            const indicatorId = $(e.target).closest('button').data('indicator-id');
            this.toggleIndicatorVisibility(indicatorId);
        });

        container.find('.edit-indicator').on('click', (e) => {
            const indicatorId = $(e.target).closest('button').data('indicator-id');
            const config = this.activeIndicators.get(indicatorId);
            this.showIndicatorSettings(indicatorId, config.params);
        });

        container.find('.remove-indicator').on('click', (e) => {
            const indicatorId = $(e.target).closest('button').data('indicator-id');
            this.removeIndicator(indicatorId);
        });
    }

    /**
     * Toggle indicator visibility
     */
    toggleIndicatorVisibility(indicatorId) {
        const config = this.activeIndicators.get(indicatorId);
        if (config) {
            config.visible = !config.visible;
            this.updateActiveIndicatorsList();

            // Reload chart to reflect visibility change
            this.loadChart();
        }
    }

    /**
     * Remove indicator from chart
     */
    removeIndicator(indicatorId) {
        if (!confirm('Remove this indicator?')) return;

        try {
            // Remove from active indicators
            this.activeIndicators.delete(indicatorId);

            // Remove traces from chart
            this.removeIndicatorTraces(indicatorId);

            // Update UI
            this.updateActiveIndicatorsList();

            console.log(`✅ Removed indicator: ${indicatorId}`);

        } catch (error) {
            console.error('❌ Error removing indicator:', error);
            this.showError('Failed to remove indicator');
        }
    }

    /**
     * Refresh single indicator on chart
     */
    async refreshIndicator(indicatorId) {
        const config = this.activeIndicators.get(indicatorId);
        if (!config || !config.visible) return;

        try {
            // Build parameter query string
            const paramString = Object.entries(config.params)
                .map(([key, value]) => `${key}=${encodeURIComponent(value)}`)
                .join('&');

            const url = `/api/indicator/${indicatorId}/calculate?symbol=${this.currentSymbol}&interval=${this.currentTimeframe}&limit=500&${paramString}`;

            const response = await fetch(url);
            const data = await response.json();

            if (!data.success) {
                throw new Error(data.error || 'Failed to calculate indicator');
            }

            // Add indicator plot to existing chart
            this.addIndicatorToChart(data);

        } catch (error) {
            console.error(`❌ Error refreshing indicator ${indicatorId}:`, error);
        }
    }

    /**
     * Refresh all active indicators
     */
    async refreshAllIndicators() {
        const promises = Array.from(this.activeIndicators.keys()).map(indicatorId =>
            this.refreshIndicator(indicatorId)
        );

        try {
            await Promise.all(promises);
        } catch (error) {
            console.error('❌ Error refreshing indicators:', error);
        }
    }

    /**
     * Add indicator plot data to existing chart
     */
    addIndicatorToChart(indicatorResults) {
        //this.debugIndicatorData(indicatorData);
        try {
            const plotlyDiv = document.getElementById('trading-chart');

            if (!plotlyDiv.data) {
                console.warn('Chart not initialized yet');
                return;
            }

            const indicators = Array.isArray(indicatorResults) ? indicatorResults : [indicatorResults];

            const overlayTraces = [];
            const subplotTraces = [];
            const layoutUpdates = {};
            let subplotCount = this.getExistingSubplotCount();

            indicators.forEach(indicatorData => {
                if (!indicatorData || !indicatorData.plot_data) {
                    console.warn('Invalid indicator data:', indicatorData);
                    return;
                }

                const plotData = indicatorData.plot_data;
                const indicatorName = indicatorData.indicator_name;
                console.log(`Processing indicator: ${indicatorName}`, plotData);

                if (plotData && plotData.data && Array.isArray(plotData.data)) {
                    const isSubplotIndicator = this.isSubplotIndicator(indicatorName);

                    if (isSubplotIndicator) {
                        subplotCount++;
                        const yAxisName = `y${subplotCount + 2}`; // y3, y4, etc.

                        // Add y-axis configuration
                        layoutUpdates[`yaxis${subplotCount + 2}`] = this.getSubplotAxisConfig(subplotCount, indicatorName);

                        plotData.data.forEach(trace => {
                            // Format timestamps
                            if (trace.x && Array.isArray(trace.x)) {
                                trace.x = trace.x.map(timestamp =>
                                    timestamp instanceof Date ? timestamp : new Date(timestamp)
                                );
                            }

                            // Apply styling
                            if (trace.line) {
                                trace.line.width = trace.line.width || 2;
                            }

                            // Assign to subplot
                            trace.yaxis = yAxisName;
                            subplotTraces.push(trace);
                            console.log(`Added ${indicatorName} to subplot with axis: ${yAxisName}`);
                        });
                    } else {
                        // Overlay traces
                        plotData.data.forEach(trace => {
                            if (trace.x && Array.isArray(trace.x)) {
                                trace.x = trace.x.map(timestamp =>
                                    timestamp instanceof Date ? timestamp : new Date(timestamp)
                                );
                            }
                            trace.yaxis = 'y';
                            overlayTraces.push(trace);
                        });
                    }
                }
            });

            // Add traces
            const allTraces = [...overlayTraces, ...subplotTraces];
            if (allTraces.length > 0) {
                Plotly.addTraces(plotlyDiv, allTraces);
                console.log(`✅ Added ${allTraces.length} indicator traces`);
            }

            // Update layout
            if (Object.keys(layoutUpdates).length > 0) {
                this.updateLayoutForSubplots(plotlyDiv, subplotCount, layoutUpdates);
            }

        } catch (error) {
            console.error('❌ Error adding indicators to chart:', error);
        }
    }

    /**
     * Get existing subplot count
     */
    getExistingSubplotCount() {
        const plotlyDiv = document.getElementById('trading-chart');
        if (!plotlyDiv.layout) return 0;

        // Count existing y-axes (excluding y and y2 which are main chart and volume)
        const yAxes = Object.keys(plotlyDiv.layout).filter(key => key.startsWith('yaxis') && key !== 'yaxis' && key !== 'yaxis2');
        return yAxes.length;
    }

    /**
     * Remove indicator traces from chart
     */
    removeIndicatorTraces(indicatorId) {
        try {
            const plotlyDiv = document.getElementById('trading-chart');
            const traceInfo = this.indicatorTraces.get(indicatorId);

            if (!traceInfo) return;

            // Remove traces in reverse order
            const indices = traceInfo.traceIndices.sort((a, b) => b - a);
            for (const index of indices) {
                if (index < plotlyDiv.data.length) {
                    Plotly.deleteTraces(plotlyDiv, index);
                }
            }

            // Update remaining trace indices
            this.updateTraceIndices(indicatorId);
            this.indicatorTraces.delete(indicatorId);

        } catch (error) {
            console.error('❌ Error removing indicator traces:', error);
        }
    }

    /**
     * Update trace indices after removal
     */
    updateTraceIndices(removedIndicatorId) {
        const removedInfo = this.indicatorTraces.get(removedIndicatorId);
        if (!removedInfo) return;

        const removedIndices = removedInfo.traceIndices;
        const minRemovedIndex = Math.min(...removedIndices);

        // Update indices for remaining indicators
        this.indicatorTraces.forEach((info, id) => {
            if (id !== removedIndicatorId) {
                info.traceIndices = info.traceIndices.map(index => {
                    return index > minRemovedIndex ? index - removedIndices.length : index;
                });
            }
        });
    }

    /**
     * Start live data updates
     */
    startLiveData() {
        if (this.isLiveMode) return;

        this.isLiveMode = true;
        $('#start-live-data').prop('disabled', true);
        $('#pause-live-data').prop('disabled', false);

        // Update every 60 seconds instead of 30
        this.pollingInterval = setInterval(() => {
            if (!this.isLoading) { // Only update if not currently loading
                this.loadChart();
            }
        }, 60000);

        console.log('📈 Started live data updates');
    }

    /**
     * Refresh entire chart with latest data and all indicators
     */
    async refreshEntireChart() {
        try {
            console.log('🔄 Refreshing entire chart...');

            // Simply reload the chart - this will get fresh data and recalculate all indicators
            await this.loadChart();

            console.log('✅ Chart refreshed successfully');
        } catch (error) {
            console.error('❌ Error refreshing chart:', error);
        }
    }

    /**
     * Pause live data updates
     */
    pauseLiveData() {
        if (!this.isLiveMode) return;

        this.isLiveMode = false;
        $('#start-live-data').prop('disabled', false);
        $('#pause-live-data').prop('disabled', true);

        // Unsubscribe from WebSocket updates
//        this.socket.emit('unsubscribe_symbol', {
//            symbol: this.currentSymbol,
//            interval: this.currentTimeframe
//        });

        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }

        console.log('⏸️ Paused live data updates');
    }

    /**
     * Toggle fullscreen mode
     */
    toggleFullscreen() {
        const chartElement = document.getElementById('trading-chart');

        if (!document.fullscreenElement) {
            chartElement.requestFullscreen();
        } else {
            document.exitFullscreen();
        }
    }

    /**
     * Search symbols (placeholder for future implementation)
     */
    searchSymbols(query) {
        // Clear previous timeout
        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
        }

        // Don't search for very short queries
        if (query.length < 2) {
            this.clearSearchResults();
            return;
        }

        // Debounce search
        this.searchTimeout = setTimeout(async () => {
            try {
                const response = await fetch(`/api/search-symbols?q=${encodeURIComponent(query)}`);
                const data = await response.json();

                if (data.success) {
                    this.displaySearchResults(data.symbols);
                }
            } catch (error) {
                console.error('❌ Error searching symbols:', error);
            }
        }, 300);
    }

    /**
     * Display search results
     */
    displaySearchResults(symbols) {
        const searchInput = $('#symbol-search');
        const resultsContainer = $('#search-results');

        // Remove existing results
        resultsContainer.remove();

        if (symbols.length === 0) return;

        // Create results dropdown
        const resultsHtml = `
            <div id="search-results" class="position-absolute bg-dark border rounded mt-1 w-100" style="z-index: 1000;">
                ${symbols.slice(0, 10).map(symbol => `
                    <div class="search-result-item p-2 border-bottom cursor-pointer" data-symbol="${symbol.symbol}">
                        <div class="fw-semibold">${symbol.symbol}</div>
                        <small class="text-muted">${symbol.name || ''}</small>
                    </div>
                `).join('')}
            </div>
        `;

    searchInput.parent().css('position', 'relative').append(resultsHtml);

    // Add click handlers
    $('.search-result-item').on('click', (e) => {
        const symbol = $(e.currentTarget).data('symbol');
        this.selectSymbol(symbol);
        this.clearSearchResults();
    });
}

    /**
     * Select a symbol from search
     */
    selectSymbol(symbol) {
        $('#active-symbol').val(symbol);
        $('#symbol-search').val('');
        this.currentSymbol = symbol;
        this.updateChartTitle();
        this.loadChart();
    }

    /**
     * Clear search results
     */
    clearSearchResults() {
        $('#search-results').remove();
    }

    /**
     * Format parameter name for display
     */
    formatParameterName(name) {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    /**
     * Error handling helper
     */
    showError(message) {
        console.error('Error:', message);

        // Create toast notification instead of alert
        const toast = $(`
            <div class="toast position-fixed top-0 end-0 m-3" style="z-index: 9999;">
                <div class="toast-header bg-danger text-white">
                    <strong class="me-auto">Error</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast"></button>
                </div>
                <div class="toast-body bg-dark text-white">
                    ${message}
                </div>
            </div>
        `);

        $('body').append(toast);
        const bsToast = new bootstrap.Toast(toast[0]);
        bsToast.show();

        // Auto remove after 5 seconds
        setTimeout(() => toast.remove(), 5000);
    }


    /**
     * Initialize WebSocket connection for real-time data
     */
    initializeWebSocket() {
        // Initialize Socket.IO connection
        this.socket = io();

        this.socket.on('connect', () => {
            console.log('🔌 WebSocket connected');
            this.updateConnectionStatus('connected');
        });

        this.socket.on('disconnect', () => {
            console.log('🔌 WebSocket disconnected');
            this.updateConnectionStatus('disconnected');
        });

        this.socket.on('price_update', (data) => {
            this.handlePriceUpdate(data);
        });

        this.socket.on('connection_status', (data) => {
            console.log('📡 Connection status:', data.status);
        });
    }

    /**
     * Handle real-time price updates
     */
    handlePriceUpdate(data) {
        if (data.symbol === this.currentSymbol && data.interval === this.currentTimeframe) {
            console.log('📈 Received price update:', data);

            // Update price ticker
            this.updatePriceTickerRealtime(data);

            // Update chart if in live mode
            if (this.isLiveMode) {
                this.updateChartWithNewCandle(data);
            }
        }
    }

    /**
     * Update price ticker with real-time data
     */
    updatePriceTickerRealtime(data) {
        const currentPrice = data.close;
        const previousPrice = this.lastPrice || currentPrice;
        const priceChange = currentPrice - previousPrice;
        const priceChangePercent = ((priceChange / previousPrice) * 100);

        $('#current-price').text(`$${currentPrice.toLocaleString()}`);

        const changeElement = $('#price-change');
        const changeText = `${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)} (${priceChangePercent.toFixed(2)}%)`;
        changeElement.text(changeText);
        changeElement.removeClass('positive negative');
        changeElement.addClass(priceChange >= 0 ? 'positive' : 'negative');

        this.lastPrice = currentPrice;
    }

    /**
     * Update chart with new real-time candle
     */
    updateChartWithNewCandle(data) {
        try {
            const plotlyDiv = document.getElementById('trading-chart');

            if (!plotlyDiv.data) return;

            // Find the candlestick trace (usually index 0)
            const candlestickTraceIndex = 0;

            // Update the last candle or add new one
            const update = {
                x: [[data.timestamp]],
                open: [[data.open]],
                high: [[data.high]],
                low: [[data.low]],
                close: [[data.close]]
            };

            Plotly.extendTraces(plotlyDiv, update, [candlestickTraceIndex]);

            // Keep only last 1000 candles to prevent memory issues
            if (plotlyDiv.data[candlestickTraceIndex].x.length > 1000) {
                const deleteCount = plotlyDiv.data[candlestickTraceIndex].x.length - 1000;
                Plotly.deleteTraces(plotlyDiv, Array.from({length: deleteCount}, (_, i) => i));
            }

        } catch (error) {
            console.error('❌ Error updating chart with new candle:', error);
        }
    }
}

// Initialize dashboard when DOM is ready
$(document).ready(() => {
    console.log('🎯 Initializing Trading Dashboard Application...');
    window.tradingDashboard = new TradingDashboard();
});