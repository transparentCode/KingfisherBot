// Indicators handling module
const IndicatorModule = (function() {
    // Cache for registered indicators
    let indicators = null;

    // Fetch all registered indicators
    async function loadRegisteredIndicators() {
        if (indicators !== null) return indicators;

        try {
            const response = await fetch('/api/indicators');
            const data = await response.json();

            if (data.success) {
                indicators = data.indicators;
                return indicators;
            } else {
                console.error('Failed to load indicators:', data.error);
                return [];
            }
        } catch (error) {
            console.error('Error fetching indicators:', error);
            return [];
        }
    }

    // Generate HTML for indicator selection
    async function populateIndicatorSelector(selectorId) {
        const indicators = await loadRegisteredIndicators();
        const $selector = $(`#${selectorId}`);

        $selector.empty();

        indicators.forEach(indicator => {
            $selector.append(`
                <a href="#" class="list-group-item list-group-item-action indicator-selector" data-indicator="${indicator.id}">
                    <div class="d-flex w-100 justify-content-between">
                        <h6 class="mb-1">${indicator.name}</h6>
                        <small><i class="fas fa-chart-line"></i></small>
                    </div>
                    <small class="text-muted">${indicator.description}</small>
                </a>
            `);
        });

        // Attach click event handler
        $('.indicator-selector').click(function(e) {
            e.preventDefault();
            $('.indicator-selector').removeClass('active');
            $(this).addClass('active');

            // Load parameters for selected indicator
            const indicatorId = $(this).data('indicator');
            loadIndicatorParams(indicatorId);
        });
    }

    // Load and display parameters for a specific indicator
    async function loadIndicatorParams(indicatorId) {
        try {
            // Hide all parameter sections
            $('.indicator-params').hide();

            // Check if we already have parameters section
            const paramSectionId = `params_${indicatorId}`;
            if ($(`#${paramSectionId}`).length > 0) {
                $(`#${paramSectionId}`).show();
                return;
            }

            // Fetch parameters from API
            const response = await fetch(`/api/indicator/params/${indicatorId}`);
            const data = await response.json();

            if (!data.success) {
                console.error('Failed to load parameters:', data.error);
                return;
            }

            // Create parameters section
            const $paramsSection = $(`<div id="${paramSectionId}" class="indicator-params mt-4"></div>`);
            $paramsSection.append(`<h6 class="mb-3">${data.indicator_name} Parameters</h6>`);

            // Add parameter inputs based on parameter types
            Object.entries(data.parameters).forEach(([name, info]) => {
                let inputHtml = '';

                if (info.type === 'bool') {
                    // Boolean checkbox
                    inputHtml = `
                        <div class="form-check mb-3">
                            <input class="form-check-input indicator-param" type="checkbox"
                                id="${indicatorId}_${name}" data-param="${name}"
                                ${info.default ? 'checked' : ''}>
                            <label class="form-check-label" for="${indicatorId}_${name}">
                                ${formatParameterName(name)}
                            </label>
                        </div>
                    `;
                } else if (info.type === 'str') {
                    // String dropdown if we have options, otherwise text input
                    if (name.toLowerCase().includes('method') || name.toLowerCase().includes('type')) {
                        // For methods/types, create select dropdown with common options
                        let options = '';

                        // Add common options based on parameter name
                        if (name.toLowerCase().includes('method')) {
                            options = `
                                <option value="atr" ${info.default === 'atr' ? 'selected' : ''}>ATR-based</option>
                                <option value="stdev" ${info.default === 'stdev' ? 'selected' : ''}>Standard Deviation</option>
                                <option value="linreg" ${info.default === 'linreg' ? 'selected' : ''}>Linear Regression</option>
                            `;
                        } else {
                            // Add a default option
                            options = `<option value="${info.default}">${info.default}</option>`;
                        }

                        inputHtml = `
                            <div class="mb-3">
                                <label for="${indicatorId}_${name}">${formatParameterName(name)}</label>
                                <select class="form-select indicator-param" id="${indicatorId}_${name}" data-param="${name}">
                                    ${options}
                                </select>
                            </div>
                        `;
                    } else {
                        // Text input
                        inputHtml = `
                            <div class="mb-3">
                                <label for="${indicatorId}_${name}">${formatParameterName(name)}</label>
                                <input type="text" class="form-control indicator-param"
                                    id="${indicatorId}_${name}" data-param="${name}" value="${info.default}">
                            </div>
                        `;
                    }
                } else if (info.type === 'int') {
                    // For integer parameters, use a slider with a value display
                    // Guess reasonable min/max values based on parameter name
                    let min = 1, max = 200;

                    if (name.toLowerCase().includes('period') || name.toLowerCase().includes('length')) {
                        min = 1;
                        max = 50;
                    }

                    inputHtml = `
                        <div class="mb-3">
                            <label for="${indicatorId}_${name}" class="form-label d-flex justify-content-between">
                                <span>${formatParameterName(name)}</span>
                                <span class="badge bg-primary" id="${indicatorId}_${name}_value">${info.default}</span>
                            </label>
                            <input type="range" class="form-range indicator-param"
                                id="${indicatorId}_${name}" data-param="${name}"
                                value="${info.default}" min="${min}" max="${max}" step="1">
                            <div class="d-flex justify-content-between">
                                <small class="text-muted">${min}</small>
                                <small class="text-muted">${max}</small>
                            </div>
                        </div>
                    `;
                } else if (info.type === 'float') {
                    // For float parameters, use a slider with value display
                    let min = 0.1, max = 5.0, step = 0.1;

                    inputHtml = `
                        <div class="mb-3">
                            <label for="${indicatorId}_${name}" class="form-label d-flex justify-content-between">
                                <span>${formatParameterName(name)}</span>
                                <span class="badge bg-secondary" id="${indicatorId}_${name}_value">${info.default.toFixed(1)}</span>
                            </label>
                            <input type="range" class="form-range indicator-param"
                                id="${indicatorId}_${name}" data-param="${name}"
                                value="${info.default}" min="${min}" max="${max}" step="${step}">
                            <div class="d-flex justify-content-between">
                                <small class="text-muted">${min}</small>
                                <small class="text-muted">${max}</small>
                            </div>
                        </div>
                    `;
                } else {
                    // Unknown type - generic text input
                    inputHtml = `
                        <div class="mb-3">
                            <label for="${indicatorId}_${name}">${formatParameterName(name)}</label>
                            <input type="text" class="form-control indicator-param"
                                id="${indicatorId}_${name}" data-param="${name}" value="${info.default}">
                        </div>
                    `;
                }

                $paramsSection.append(inputHtml);
            });

            // Add section to DOM
            $('#indicatorParamsContainer').append($paramsSection);

            // Add event handlers for sliders
            $(`#${paramSectionId} input[type="range"]`).on('input', function() {
                const id = $(this).attr('id');
                $(`#${id}_value`).text($(this).val());
            });
        } catch (error) {
            console.error('Error loading indicator parameters:', error);
        }
    }

    // Format parameter name for display (convert snake_case to Title Case)
    function formatParameterName(name) {
        return name
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
    }

    // Generate indicator plot
    async function generateIndicatorPlot() {
        const indicatorId = $('.indicator-selector.active').data('indicator');
        if (!indicatorId) {
            alert('Please select an indicator first');
            return;
        }

        // Collect general parameters
        const symbol = $('#vis_symbol').val();
        const interval = $('#vis_interval').val();
        const lookback = $('#vis_lookback').val();

        // Show loading
        const loadingModal = bootstrap.Modal.getOrCreateInstance(document.getElementById('loadingModal'));
        $('#loadingMessage').text('Generating indicator visualization...');
        loadingModal.show();

        // Collect indicator-specific parameters
        const paramSectionId = `params_${indicatorId}`;
        let paramQueryString = '';

        $(`#${paramSectionId} .indicator-param`).each(function() {
            const param = $(this).data('param');
            let value;

            if ($(this).attr('type') === 'checkbox') {
                value = $(this).is(':checked');
            } else {
                value = $(this).val();
            }

            paramQueryString += `&${param}=${value}`;
        });

        try {
            // Make API call
            const response = await fetch(`/api/indicator/${indicatorId}?symbol=${symbol}&interval=${interval}&lookback_days=${lookback}${paramQueryString}`);
            const data = await response.json();

            loadingModal.hide();

            if (data.success) {
                // Clear placeholder
                $('#indicatorPlotDiv').empty();

                // Render plot
                Plotly.newPlot('indicatorPlotDiv', data.plot_data.data, data.plot_data.layout, {
                    responsive: true,
                    displayModeBar: true,
                    modeBarButtonsToAdd: ['drawline', 'drawopenpath', 'eraseshape'],
                    modeBarButtonsToRemove: ['lasso2d', 'select2d']
                });

                // Update insights if needed
                updateIndicatorInsights(indicatorId, data);
            } else {
                $('#indicatorPlotDiv').html(`
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle"></i>
                        Error generating plot: ${data.error}
                    </div>
                `);
            }
        } catch (error) {
            loadingModal.hide();
            $('#indicatorPlotDiv').html(`
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle"></i>
                    API request failed: ${error}
                </div>
            `);
        }
    }

    // Update indicator insights
    function updateIndicatorInsights(indicatorId, data) {
        const $insights = $('#indicatorInsights');
        $insights.empty();

        try {
            // Generic insights for all indicators
            $insights.append(`
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Symbol
                    <span class="badge bg-primary">${data.symbol}</span>
                </li>
                <li class="list-group-item d-flex justify-content-between align-items-center">
                    Timeframe
                    <span class="badge bg-info">${data.interval}</span>
                </li>
            `);

            // Get plot data for more specific insights
            const plotData = data.plot_data;

            // Add indicator-specific insights based on indicator ID
            if (indicatorId === 'trendlines') {
                // Find breakout signals
                const upBreaks = plotData.data.find(t => t.name === 'Upward Breakout');
                const downBreaks = plotData.data.find(t => t.name === 'Downward Breakout');

                const upBreakCount = upBreaks && upBreaks.x ? upBreaks.x.length : 0;
                const downBreakCount = downBreaks && downBreaks.x ? downBreaks.x.length : 0;

                // Add breakout counts
                $insights.append(`
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        Upward Breakouts
                        <span class="badge bg-success rounded-pill">${upBreakCount}</span>
                    </li>
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        Downward Breakouts
                        <span class="badge bg-danger rounded-pill">${downBreakCount}</span>
                    </li>
                `);

                // Add calculation method info
                const calcMethod = data.parameters.calculate_method;
                let methodBadge = '';

                if (calcMethod === 'atr') {
                    methodBadge = '<span class="badge bg-primary">ATR-based</span>';
                } else if (calcMethod === 'stdev') {
                    methodBadge = '<span class="badge bg-info">StdDev-based</span>';
                } else {
                    methodBadge = '<span class="badge bg-warning text-dark">LinReg-based</span>';
                }

                $insights.append(`
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        Calculation Method
                        ${methodBadge}
                    </li>
                `);
            }
            // Add other indicator types here as needed

        } catch (e) {
            console.error('Error extracting insights:', e);
            $insights.append(`
                <li class="list-group-item text-center text-muted">
                    <small>Could not extract insights from plot data</small>
                </li>
            `);
        }
    }

    // Public API
    return {
        init: function() {
            // Initialize on page load
            populateIndicatorSelector('indicatorList');

            // Attach event handler to generate button
            $('#generateIndicatorPlot').click(generateIndicatorPlot);
        }
    };
})();

// Initialize on document ready
$(document).ready(function() {
    IndicatorModule.init();
});