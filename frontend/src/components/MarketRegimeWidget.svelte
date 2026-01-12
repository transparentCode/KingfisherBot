<script>
    import { marketStore } from '../stores/marketStore.js';
    import { fade } from 'svelte/transition';
    
    $: regimeData = $marketStore.regime;
    $: loading = $marketStore.regimeLoading;
    $: regimeStr = String(regimeData?.regime ?? '');
    $: trendStrength = clamp01(Number(regimeData?.trend_strength));
    $: volatility = clamp01(Number(regimeData?.volatility));
    $: hurst = toFiniteOrNull(regimeData?.hurst);
    $: cyclePeriod = toFiniteOrNull(regimeData?.cycle_period);
    $: cyclePhase = toFiniteOrNull(regimeData?.cycle_phase);

    function formatRegime(raw) {
        if (!raw) return 'UNKNOWN';
        return raw.replace(/_/g, ' ').toUpperCase();
    }

    function clamp01(val) {
        if (!Number.isFinite(val)) return 0;
        return Math.min(1, Math.max(0, val));
    }

    function toFiniteOrNull(val) {
        const num = Number(val);
        return Number.isFinite(num) ? num : null;
    }

    function fmtPct01(val) {
        const num = Number(val);
        if (!Number.isFinite(num)) return '--';
        return `${Math.round(clamp01(num) * 100)}%`;
    }

    function fmt2(val) {
        const num = Number(val);
        return Number.isFinite(num) ? num.toFixed(2) : '--';
    }

    // --- Helpers for Tooltips & Colors ---
    const tooltips = {
        strength: "Trend Strength (0-1). >0.25 indicates significant trend.",
        volatility: "Market Volatility/Stress. High values imply risk of reversal.",
        hurst: "Hurst Exponent. >0.5: Trending, <0.5: Mean Reverting, ≈0.5: Random.",
        cycle: "Hilbert Transform Cycle. Phase indicates current wave position."
    };

    function getStrengthColor(val) {
        if (val > 0.6) return 'var(--success)';
        if (val > 0.3) return 'var(--text)';
        return 'var(--muted)';
    }

    function getVolColor(val) {
        if (val > 0.7) return 'var(--danger)';
        if (val > 0.4) return 'var(--warning)'; 
        return 'var(--text)';
    }
</script>

<div class="material-widget elevation-2">
    {#if loading}
        <div class="loading-state" in:fade>
             <div class="spinner"></div>
             <span>Analyzing Market Structure...</span>
        </div>
    {:else if regimeData}
        <!-- Section 1: Regime Identifier -->
        <div class="primary-section">
            <span class="section-label">MARKET REGIME</span>
            <div class="regime-display" 
                 style="color: {regimeStr.includes('BULL') ? 'var(--success)' : 
                                 regimeStr.includes('BEAR') ? 'var(--danger)' : 
                                 'var(--warning)'}">
                {formatRegime(regimeStr)}
            </div>
        </div>

        <div class="divider"></div>

        <!-- Section 2: Metrics Grid -->
        <div class="metrics-grid">
            
            <!-- Trend Strength -->
            <div class="metric-item tooltip" data-tip={tooltips.strength}>
                <div class="metric-header">
                    <span>STRENGTH</span>
                    <span style="color: {getStrengthColor(trendStrength)}">
                        {fmtPct01(trendStrength)}
                    </span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" 
                         style="width: {trendStrength * 100}%; 
                                background-color: {getStrengthColor(trendStrength)}">
                    </div>
                </div>
            </div>

            <!-- Volatility -->
            <div class="metric-item tooltip" data-tip={tooltips.volatility}>
                <div class="metric-header">
                    <span>VOLATILITY</span>
                    <span style="color: {getVolColor(volatility)}">
                        {fmtPct01(volatility)}
                    </span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" 
                         style="width: {volatility * 100}%; 
                                background-color: {getVolColor(volatility)}">
                    </div>
                </div>
            </div>

            <!-- Hurst / Cycle -->
            <div class="metric-mini-row">
                 <div class="mini-stat tooltip" data-tip={tooltips.hurst}>
                    <span class="label">HURST</span>
                    <span class="value">{fmt2(hurst)}</span>
                 </div>
                 <div class="mini-stat tooltip" data-tip={tooltips.cycle}>
                    <span class="label">CYCLE</span>
                    <span class="value" style="color: var(--accent)">
                        {#if cyclePeriod !== null}
                           DC:{Math.round(cyclePeriod)}
                        {:else if cyclePhase !== null}
                           Φ:{fmt2(cyclePhase)}
                        {:else}
                            --
                        {/if}
                    </span>
                 </div>
            </div>

        </div>
    {:else}
        <div class="empty-state">
            <span>-- No Data --</span>
        </div>
    {/if}
</div>

<style>
    :root {
        --warning: #f4a261;
        --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.3), 0 2px 4px -1px rgba(0, 0, 0, 0.16);
    }

    .material-widget {
        display: flex;
        align-items: center;
        background: var(--surface-1); /* Dark card bg */
        border: 1px solid var(--border);
        border-radius: 12px;
        box-shadow: var(--shadow-md);
        padding: 12px 20px;
        gap: 24px;
        min-height: 60px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .material-widget:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 12px -2px rgba(0, 0, 0, 0.4);
        border-color: var(--accent);
    }

    /* Primary Section */
    .primary-section {
        display: flex;
        flex-direction: column;
        justify-content: center;
        min-width: 110px;
    }
    .section-label {
        font-size: 0.7rem;
        color: var(--muted);
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 2px;
    }
    .regime-display {
        font-size: 1.1rem;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: 0.02em;
        text-shadow: 0 1px 2px rgba(0,0,0,0.5);
    }

    .divider {
        width: 1px;
        height: 40px;
        background: var(--border);
    }

    /* Metrics Grid */
    .metrics-grid {
        display: flex;
        gap: 20px;
        align-items: center;
    }

    .metric-item {
        display: flex;
        flex-direction: column;
        width: 100px;
        gap: 4px;
        position: relative;
        cursor: help;
    }

    .metric-header {
        display: flex;
        justify-content: space-between;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .metric-header span:first-child {
        color: var(--muted);
        font-size: 0.65rem;
    }

    .progress-track {
        height: 6px;
        width: 100%;
        background: rgba(255,255,255,0.05);
        border-radius: 3px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 3px;
        transition: width 0.5s ease-out;
    }

    /* Mini Stats */
    .metric-mini-row {
        display: flex;
        gap: 16px;
    }
    .mini-stat {
        display: flex;
        flex-direction: column;
        cursor: help;
        position: relative;
    }
    .mini-stat .label {
        font-size: 0.65rem;
        color: var(--muted);
        letter-spacing: 0.5px;
        margin-bottom: 2px;
    }
    .mini-stat .value {
        font-size: 0.9rem;
        font-weight: 700;
        color: var(--text);
        font-family: 'Roboto Mono', monospace;
    }

    /* Loading / Empty */
    .loading-state, .empty-state {
        display: flex;
        align-items: center;
        gap: 10px;
        color: var(--muted);
        font-size: 0.9rem;
        font-style: italic;
    }
    .spinner {
        width: 16px; 
        height: 16px; 
        border: 2px solid var(--accent); 
        border-top-color: transparent; 
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* Tooltip Implementation (CSS Only) */
    .tooltip::before, .tooltip::after {
        opacity: 0;
        visibility: hidden;
        transition: 0.2s opacity ease;
        pointer-events: none;
        position: absolute;
        z-index: 10;
        transform: translateX(-50%);
        left: 50%;
    }

    .tooltip:hover::before, .tooltip:hover::after {
        opacity: 1;
        visibility: visible;
    }

    /* The Text */
    .tooltip::after {
        content: attr(data-tip);
        background: #333;
        color: #fff;
        font-size: 11px;
        padding: 6px 10px;
        border-radius: 4px;
        white-space: nowrap;
        bottom: 110%; /* Place above */
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }

    /* The Arrow */
    .tooltip::before {
        content: "";
        border: 6px solid transparent;
        border-top-color: #333;
        bottom: 90%;
    }
    
    @media (max-width: 800px) {
        .material-widget {
            flex-wrap: wrap;
            height: auto;
            padding: 10px;
            gap: 12px;
        }
        .metrics-grid {
            flex-wrap: wrap;
            gap: 12px;
        }
        .divider { display: none; }
    }
</style>
