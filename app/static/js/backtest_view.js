$(document).ready(function() {

    // Add TradingView widget script
      const script = document.createElement('script');
        script.src = 'https://s3.tradingview.com/tv.js';
        script.async = true;
        script.onload = function() {
          // Initialize all widgets
          initTradingViewWidget($('#symbol').val());
          initHeaderTradingViewWidget($('#symbol').val());
          initNavbarTradingViewWidget($('#symbol').val());
        };
        document.head.appendChild(script);

        // Set up navbar symbol selector
        $('#navbar_symbol_selector').change(function() {
          initNavbarTradingViewWidget($(this).val());
        });

        // Set up refresh button
        $('#refreshNavbarChart').click(function() {
          initNavbarTradingViewWidget($('#navbar_symbol_selector').val());
        });

        // Update all charts when main symbol changes
        $('#symbol').change(function() {
          const symbol = $(this).val();
          initTradingViewWidget(symbol);
          initHeaderTradingViewWidget(symbol);
          // Optionally sync the navbar chart too:
          $('#navbar_symbol_selector').val(symbol);
          initNavbarTradingViewWidget(symbol);
        });

    // Strategy parameter toggle
    $('#strategy_id').change(function() {
        $('.strategy-params').hide();
        $('#' + $(this).val() + '_params').show();
    }).trigger('change');

    // Leverage toggle
    $('#leverage_switch').change(function() {
        if ($(this).is(':checked')) {
            $('#leverage_options').slideDown();
        } else {
            $('#leverage_options').slideUp();
        }
    });

    // Update leverage value display
    $('#leverage').on('input', function() {
        $('#leverageValue').text($(this).val() + 'x');
    });

    // Format number as percentage
    function formatPercent(value) {
        return (value * 100).toFixed(2) + '%';
    }

    // Format number with color based on value
    function formatColoredValue(value, invert = false) {
        let formattedValue = '';
        let className = '';

        if (value > 0) {
            className = invert ? 'negative' : 'positive';
            formattedValue = '+' + formatPercent(value);
        } else if (value < 0) {
            className = invert ? 'positive' : 'negative';
            formattedValue = formatPercent(value);
        } else {
            formattedValue = '0.00%';
        }

        return '<span class="' + className + '">' + formattedValue + '</span>';
    }

    // Store backtest history
    let backtestHistory = [];

    // Run backtest
    $('#runBacktest').click(function() {
        const strategyId = $('#strategy_id').val();

        // Collect strategy parameters
        let strategyParams = {};
        $('#' + strategyId + '_params .strategy-param').each(function() {
            const param = $(this).data('param');
            const value = $(this).val();
            strategyParams[param] = parseFloat(value);
        });

        const formData = {
            symbol: $('#symbol').val(),
            interval: $('#interval').val(),
            lookback_days: parseInt($('#lookback_days').val()),
            strategy_id: strategyId,
            strategy_params: strategyParams
        };

        if ($('#leverage_switch').is(':checked')) {
            formData.leverage = parseInt($('#leverage').val());
        }

        // Show loading modal with random messages
        const loadingMessages = [
            "Analyzing historical price patterns...",
            "Calculating trade signals...",
            "Optimizing strategy parameters...",
            "Computing performance metrics...",
            "Generating visualization data...",
            "Backtesting strategy performance..."
        ];

        const loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        loadingModal.show();

        let msgIndex = 0;
        const messageInterval = setInterval(() => {
            msgIndex = (msgIndex + 1) % loadingMessages.length;
            $('#loadingMessage').text(loadingMessages[msgIndex]);
        }, 1500);

        // Run the backtest
        $.ajax({
            url: '/api/backtest',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify(formData),
            success: function(response) {
                // Clear loading interval
                clearInterval(messageInterval);

                // Update backtest info
                const now = new Date();
                const timeStr = now.toLocaleTimeString();
                $('#backtestInfo').text(
                    `${formData.symbol} (${formData.interval}) - ${$('#strategy_id option:selected').text()} - ${timeStr}`
                );

                // Add to history
                const historyItem = {
                    symbol: formData.symbol,
                    interval: formData.interval,
                    strategy: $('#strategy_id option:selected').text(),
                    timestamp: now,
                    return: response.metrics.total_return
                };
                backtestHistory.unshift(historyItem);
                updateHistoryTab();

                // Update metrics
                updateMetrics(response.metrics);

                // Fetch and display the plot
                $.ajax({
                    url: response.plot_endpoint,
                    type: 'GET',
                    success: function(plotResponse) {
                        try {
                            // Hide loading modal
                            loadingModal.hide();

                            // Parse the JSON plot data and render it
                            const plotData = plotResponse.plot_data;
                            Plotly.newPlot('plotDiv', plotData.data, plotData.layout, {
                                responsive: true,
                                displayModeBar: true,
                                modeBarButtonsToAdd: [
                                    'drawopenpath',
                                    'eraseshape'
                                ],
                                modeBarButtonsToRemove: [
                                    'lasso2d',
                                    'select2d'
                                ]
                            });

                            // Add custom interactivity
                            document.getElementById('plotDiv').on('plotly_click', function(data) {
                                // Show point details on click
                                if (data.points.length > 0) {
                                    const point = data.points[0];
                                    if (point.data.name === 'Entries' || point.data.name === 'Exits') {
                                        const date = new Date(point.x).toLocaleString();
                                        const price = point.y.toFixed(2);
                                        const type = point.data.name;

                                        // Display a small tooltip
                                        Plotly.Fx.hover('plotDiv', {
                                            curveNumber: point.curveNumber,
                                            pointNumber: point.pointNumber
                                        });

                                        // You could also display more details in a modal or card
                                        console.log(`${type} at ${date}: $${price}`);
                                    }
                                }
                            });
                        } catch (e) {
                            console.error('Error rendering plot:', e);
                            $('#plotDiv').html(`
                                        <div class="alert alert-danger">
                                            <i class="fas fa-chart-line"></i>
                                            Error rendering the backtest chart: ${e.message}
                                        </div>
                                    `);
                            loadingModal.hide();
                        }
                    },
                    error: function(error) {
                        console.error('Error fetching plot:', error);
                        $('#plotDiv').html('<div class="alert alert-danger">Error fetching plot</div>');
                        loadingModal.hide();
                    }
                });
            },
            error: function(error) {
                clearInterval(messageInterval);
                console.error('Error running backtest:', error);
                loadingModal.hide();

                let errorMsg = 'Unknown error occurred';
                if (error.responseJSON && error.responseJSON.error) {
                    errorMsg = error.responseJSON.error;
                }

                $('#plotDiv').html(`
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-triangle"></i>
                                Error running backtest: ${errorMsg}
                            </div>
                        `);
            }
        });
    });


    // View strategy details
    $('#viewStrategy').click(function() {
        const strategyId = $('#strategy_id').val();
        const symbol = $('#symbol').val();
        const interval = $('#interval').val();
        viewStrategyDetails(strategyId, symbol, interval);
    });

    // Update metrics display
    function updateMetrics(metrics) {
        // Format and display each metric
        $('#totalReturn').html(formatColoredValue(metrics.total_return));
        $('#maxDrawdown').html(formatColoredValue(metrics.max_drawdown, true));
        $('#winRate').text(formatPercent(metrics.win_rate));
        $('#profitFactor').text(metrics.profit_factor ? metrics.profit_factor.toFixed(2) : '—');
        $('#sharpeRatio').text(metrics.sharpe_ratio ? metrics.sharpe_ratio.toFixed(2) : '—');
        $('#sortinoRatio').text(metrics.sortino_ratio ? metrics.sortino_ratio.toFixed(2) : '—');
        $('#numTrades').text(metrics.num_trades);
        $('#avgDuration').text(metrics.avg_trade_duration || '—');
    }


    // Update history tab
    function updateHistoryTab() {
        const $historyTab = $('#historyTab');
        $historyTab.empty();

        if (backtestHistory.length === 0) {
            $historyTab.append('<li class="list-group-item text-center text-muted">No recent backtests</li>');
            return;
        }

        // Show last 10 backtests
        backtestHistory.slice(0, 10).forEach((item, index) => {
            const timeStr = item.timestamp.toLocaleTimeString();
            const returnClass = item.return >= 0 ? 'positive' : 'negative';
            const returnStr = formatPercent(item.return);

            $historyTab.append(`
                        <li class="list-group-item py-2">
                            <div class="d-flex justify-content-between align-items-center">
                                <div>
                                    <strong>${item.symbol}</strong>
                                    <small class="text-muted">${item.interval}</small>
                                </div>
                                <span class="${returnClass}">${returnStr}</span>
                            </div>
                            <div class="d-flex justify-content-between">
                                <small class="text-muted">${item.strategy}</small>
                                <small class="text-muted">${timeStr}</small>
                            </div>
                        </li>
                    `);
        });
    }

    // Tab switching functionality
    $('#strategyVisTab').click(function(e) {
        e.preventDefault();
        $('.container > div').hide();
        $('#strategyVisSection').show();

        // Update active nav link
        $('.navbar-nav .nav-link').removeClass('active');
        $(this).addClass('active');
    });

    // Return to backtest tab
    $('.navbar-nav .nav-link').not('#strategyVisTab').click(function(e) {
        if ($(this).attr('href') === '/backtest/view') {
            e.preventDefault();
            $('#strategyVisSection').hide();
            $('.container > div').not('#strategyVisSection').show();

            // Update active nav link
            $('.navbar-nav .nav-link').removeClass('active');
            $(this).addClass('active');
        }
    });

    // Select active strategy
    $('.strat-selector').click(function(e) {
        e.preventDefault();
        $('.strat-selector').removeClass('active');
        $(this).addClass('active');

        // Show corresponding params
        const strategy = $(this).data('strategy');
        $('.strategy-params').hide();
        $(`#vis_${strategy}_params`).show();
    });

    // SMA strategy parameter handlers
    $('#vis_fast_period').on('input', function() {
        $('#vis_fast_period_value').text($(this).val());
        updateVisSMADifference();
    });

    $('#vis_slow_period').on('input', function() {
        $('#vis_slow_period_value').text($(this).val());
        updateVisSMADifference();
    });

    function updateVisSMADifference() {
        const fastPeriod = parseInt($('#vis_fast_period').val());
        const slowPeriod = parseInt($('#vis_slow_period').val());
        const difference = slowPeriod - fastPeriod;
        $('#vis_sma_difference').text(difference);
    }

    $('#goToVisualization').click(function(e) {
        e.preventDefault();
        $('#strategyVisTab').click();

        // Transfer current backtest form values to visualization form
        $('#vis_symbol').val($('#symbol').val());
        $('#vis_interval').val($('#interval').val());
        $('#vis_lookback').val($('#lookback_days').val());

        const strategyId = $('#strategy_id').val();
        $(`.strat-selector[data-strategy="${strategyId}"]`).click();

        // Transfer strategy parameters
        $(`#${strategyId}_params .strategy-param`).each(function() {
            const paramName = $(this).data('param');
            const value = $(this).val();
            $(`#vis_${strategyId}_params [data-param="${paramName}"]`).val(value).trigger('input');
        });
    });

    // Initialize on page load
    $(document).ready(function() {
        $('#strategyVisSection').hide();
        updateVisSMADifference();

        // Default to backtest tab
        $('.navbar-nav .nav-link[href="/backtest/view"]').addClass('active');
    });

    // Generate strategy plot
    $('#generateStrategyPlot').click(function() {
        // Get selected strategy
        const strategy = $('.strat-selector.active').data('strategy');
        const symbol = $('#vis_symbol').val();
        const interval = $('#vis_interval').val();
        const lookback = $('#vis_lookback').val();

        // Show loading
        const loadingModal = bootstrap.Modal.getOrCreateInstance(document.getElementById('loadingModal'));
        $('#loadingMessage').text('Generating strategy visualization...');
        loadingModal.show();

        // Collect strategy parameters
        let paramQueryString = '';
        $(`#vis_${strategy}_params .strategy-param`).each(function() {
            const param = $(this).data('param');
            const value = $(this).val();
            paramQueryString += `&param_${param}=${value}`;
        });

        // Make API call
        $.ajax({
            url: `/api/strategy/plot?strategy_id=${strategy}&symbol=${symbol}&interval=${interval}&lookback_days=${lookback}${paramQueryString}`,
            type: 'GET',
            success: function(response) {
                loadingModal.hide();

                if (response.success) {
                    // Clear placeholder
                    $('#strategyPlotDiv').empty();

                    // Render plot
                    Plotly.newPlot('strategyPlotDiv', response.plot_data.data, response.plot_data.layout, {
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
                        modeBarButtonsToRemove: ['lasso2d', 'select2d']
                    });

                    // Update insights
                    updateStrategyInsights(strategy, response);
                } else {
                    $('#strategyPlotDiv').html(`
                                <div class="alert alert-danger">
                                    <i class="fas fa-exclamation-triangle"></i>
                                    Error generating plot: ${response.error}
                                </div>
                            `);
                }
            },
            error: function(error) {
                loadingModal.hide();
                $('#strategyPlotDiv').html(`
                            <div class="alert alert-danger">
                                <i class="fas fa-exclamation-triangle"></i>
                                API request failed: ${error.statusText}
                            </div>
                        `);
            }
        });
    });

    // Export strategy plot
    $('#exportStrategyPlot').click(function() {
        Plotly.downloadImage('strategyPlotDiv', {
            format: 'png',
            width: 1200,
            height: 800,
            filename: 'strategy_visualization'
        });
    });

    // Fullscreen button handler
    $('#fullscreenStrategyPlot').click(function() {
        const plot = document.getElementById('strategyPlotDiv');

        if (plot.requestFullscreen) {
            plot.requestFullscreen();
        } else if (plot.webkitRequestFullscreen) {
            /* Safari */
            plot.webkitRequestFullscreen();
        } else if (plot.msRequestFullscreen) {
            /* IE11 */
            plot.msRequestFullscreen();
        }
    });

    // Update strategy insights
    function updateStrategyInsights(strategy, response) {
        const $insights = $('#stratVis_insights');
        const $params = $('#stratVis_params');

        // Clear previous content
        $insights.empty();
        $params.empty();

        // Add parameters display
        $.each(response.parameters, function(key, value) {
            $params.append(`
                        <div class="badge bg-light text-dark p-2 me-2 mb-2">
                            ${key.replace('_', ' ')}: <strong>${value}</strong>
                        </div>
                    `);
        });

        // Add strategy-specific insights
        if (strategy === 'sma_confluence') {
            // Let's extract some insights from the plot data
            const plotData = response.plot_data;

            try {
                // Get the latest data points
                const priceTrace = plotData.data.find(t => t.name === 'Price');
                const fastSmaTrace = plotData.data.find(t => t.name && t.name.includes('Fast SMA'));
                const slowSmaTrace = plotData.data.find(t => t.name && t.name.includes('Slow SMA'));

                if (priceTrace && fastSmaTrace && slowSmaTrace) {
                    const lastIndex = priceTrace.close ? priceTrace.close.length - 1 : -1;

                    if (lastIndex >= 0) {
                        const price = priceTrace.close[lastIndex];
                        const fastSma = fastSmaTrace.y[lastIndex];
                        const slowSma = slowSmaTrace.y[lastIndex];

                        // Add signal analysis
                        const signalType = fastSma > slowSma ?
                            '<span class="badge bg-success">Bullish</span>' :
                            '<span class="badge bg-danger">Bearish</span>';

                        $insights.append(`
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        Current signal
                                        ${signalType}
                                    </li>
                                `);

                        // Price position
                        let pricePosition;
                        let positionBadge;

                        if (price > fastSma && price > slowSma) {
                            pricePosition = "Above both SMAs";
                            positionBadge = '<span class="badge bg-success">Strong bullish</span>';
                        } else if (price < fastSma && price < slowSma) {
                            pricePosition = "Below both SMAs";
                            positionBadge = '<span class="badge bg-danger">Strong bearish</span>';
                        } else if (price > fastSma && price < slowSma) {
                            pricePosition = "Between SMAs (above fast)";
                            positionBadge = '<span class="badge bg-warning text-dark">Mixed</span>';
                        } else {
                            pricePosition = "Between SMAs (below fast)";
                            positionBadge = '<span class="badge bg-warning text-dark">Mixed</span>';
                        }

                        $insights.append(`
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        Price position
                                        ${positionBadge}
                                    </li>
                                `);

                        // Trend strength
                        const maDiff = Math.abs(fastSma - slowSma);
                        const maAvg = (fastSma + slowSma) / 2;
                        const trendStrength = (maDiff / maAvg) * 100;

                        let strengthBadge;
                        if (trendStrength < 0.5) {
                            strengthBadge = '<span class="badge bg-secondary">Weak</span>';
                        } else if (trendStrength < 2) {
                            strengthBadge = '<span class="badge bg-primary">Moderate</span>';
                        } else {
                            strengthBadge = '<span class="badge bg-danger">Strong</span>';
                        }

                        $insights.append(`
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        Trend strength
                                        ${strengthBadge}
                                    </li>
                                `);

                        // Signal count
                        const buySignals = plotData.data.find(t => t.name === 'Buy Signal');
                        const sellSignals = plotData.data.find(t => t.name === 'Sell Signal');

                        const buyCount = buySignals ? buySignals.x.length : 0;
                        const sellCount = sellSignals ? sellSignals.x.length : 0;

                        $insights.append(`
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        Buy signals
                                        <span class="badge bg-success rounded-pill">${buyCount}</span>
                                    </li>
                                    <li class="list-group-item d-flex justify-content-between align-items-center">
                                        Sell signals
                                        <span class="badge bg-danger rounded-pill">${sellCount}</span>
                                    </li>
                                `);
                    }
                }
            } catch (e) {
                console.error('Error extracting insights:', e);
                $insights.append(`
                            <li class="list-group-item text-center text-muted">
                                <small>Could not extract insights from plot data</small>
                            </li>
                        `);
            }
        }
    }

    // Initialize history tab
    updateHistoryTab();

    function viewStrategyDetails(strategyId, symbol, interval) {
        // Show loading modal
        const loadingModal = bootstrap.Modal.getOrCreateInstance(document.getElementById('loadingModal'));
        $('#loadingMessage').text('Generating strategy visualization...');
        loadingModal.show();

        // Collect strategy parameters if available
        let paramQueryString = '';
        $('#' + strategyId + '_params .strategy-param').each(function() {
            const param = $(this).data('param');
            const value = $(this).val();
            paramQueryString += `&param_${param}=${value}`;
        });

        // Make API call to get strategy-specific plot
        $.ajax({
            url: `/api/strategy/plot?strategy_id=${strategyId}&symbol=${symbol}&interval=${interval}${paramQueryString}`,
            type: 'GET',
            success: function(response) {
                loadingModal.hide();

                if (response.success) {
                    // Create a modal to display the strategy plot
                    if (!$('#strategyDetailModal').length) {
                        $('body').append(`
                                    <div class="modal fade" id="strategyDetailModal" tabindex="-1" aria-hidden="true">
                                        <div class="modal-dialog modal-xl">
                                            <div class="modal-content">
                                                <div class="modal-header">
                                                    <h5 class="modal-title" id="strategyDetailTitle">Strategy Analysis</h5>
                                                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                                </div>
                                                <div class="modal-body">
                                                    <div id="strategyDetailPlot" style="height: 600px;"></div>
                                                    <div class="row mt-3">
                                                        <div class="col-md-12">
                                                            <h6>Strategy Parameters</h6>
                                                            <div id="strategyParams" class="d-flex flex-wrap"></div>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div class="modal-footer">
                                                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                `);
                    }

                    // Update modal title
                    $('#strategyDetailTitle').text(`${response.strategy_id.toUpperCase()} Analysis: ${response.symbol} (${response.interval})`);

                    // Display parameters
                    const $paramsDiv = $('#strategyParams');
                    $paramsDiv.empty();
                    Object.entries(response.parameters).forEach(([key, value]) => {
                        $paramsDiv.append(`
                                    <div class="badge bg-light text-dark me-2 mb-2 p-2">
                                        ${key}: <strong>${value}</strong>
                                    </div>
                                `);
                    });

                    // Render the plot
                    Plotly.newPlot('strategyDetailPlot', response.plot_data.data, response.plot_data.layout, {
                        responsive: true,
                        displayModeBar: true
                    });

                    // Show the modal
                    const strategyModal = new bootstrap.Modal(document.getElementById('strategyDetailModal'));
                    strategyModal.show();
                } else {
                    alert('Failed to generate strategy visualization: ' + response.error);
                }
            },
            error: function(error) {
                loadingModal.hide();
                console.error('Error fetching strategy plot:', error);
                alert('Error fetching strategy visualization');
            }
        });
    }

    function initTradingViewWidget(symbol = 'BTCUSDT') {
      // Format symbol for TradingView (removing USDT and adding USDT as exchange)
      const formattedSymbol = symbol.replace('USDT', '');

      new TradingView.widget({
          "width": "100%",
          "height": 250,
          "symbol": `BINANCE:${formattedSymbol}USDT`,
          "interval": "15",
          "timezone": "exchange",
          "theme": "dark",
          "style": "1",
          "locale": "en",
          "toolbar_bg": "#f1f3f6",
          "enable_publishing": false,
          "hide_top_toolbar": true,
          "hide_legend": true,
          "hide_side_toolbar": true,
          "allow_symbol_change": true,
          "save_image": false,
          "container_id": "header_tradingview_chart",
          "timezone": "Asia/Kolkata"
        });
    }

    $('#refreshTVChart').click(function() {
      initTradingViewWidget($('#symbol').val());
      initHeaderTradingViewWidget($('#symbol').val());
    });
});