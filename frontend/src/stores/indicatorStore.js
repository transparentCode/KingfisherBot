import { writable, get } from 'svelte/store';
import { marketStore } from './marketStore';
import { assetStore } from './assetStore';

// Static definition of available indicators and their routes
const STATIC_INDICATORS = [
    {
        id: 'EMA',
        name: 'Exponential Moving Average',
        category: 'Moving Averages',
        description: 'Weighted trend following',
        route: '/api/ema/calculate',
        params: { period: 20, source: 'close' },
        visible: true
    },
    {
        id: 'VolumeProfile',
        name: 'Volume Profile',
        category: 'Volume',
        description: 'Volume distribution over price levels',
        route: '/api/volume_profile/calculate',
        params: { bins: 100, lookback: 200, session_mode: false },
        visible: false
    },
    {
        id: 'VWR',
        name: 'Volume Weighted Regression',
        category: 'Trend',
        description: 'Trend analysis using volume-weighted regression',
        route: '/api/vwr/calculate',
        params: { lookback: 100, std_multiplier: 2.0 },
        visible: false
    },
    {
        id: 'RegimeMetrics',
        name: 'Regime Metrics',
        category: 'Analysis',
        description: 'Market regime classification (Hurst, Skew, Kurtosis)',
        route: '/api/regime/calculate',
        params: { hurst_lookback: 250 },
        visible: false
    }
    // RSI and Fractal are handled by dedicated components/routes in App.svelte
];

function createIndicatorStore() {
    const { subscribe, set, update } = writable({
        availableIndicators: STATIC_INDICATORS.map(ind => ({
            ...ind,
            loading: false,
            data: null,
            tvlcData: null,
            lastQueryKey: null,
            error: null
        })),
        loading: false,
        error: null
    });

    return {
        subscribe,

        // No longer fetching from backend; just resetting if needed
        fetchAvailableIndicators: async () => {
            // No-op or reset logic if needed
        },

        toggleIndicator: async (indicatorId) => {
            // First toggle the visibility state
            update(s => {
                const indicators = s.availableIndicators.map(ind => {
                    if (ind.id === indicatorId) {
                        return { ...ind, visible: !ind.visible };
                    }
                    return ind;
                });
                return { ...s, availableIndicators: indicators };
            });

            // Check if we need to fetch data; read current state after toggle
            const currentState = get(indicatorStore);
            const targetIndicator = currentState.availableIndicators.find(i => i.id === indicatorId);

            // When toggled ON, always fetch fresh data so the chart reliably restores the overlay.
            if (!targetIndicator || !targetIndicator.visible) return;
            if (targetIndicator.loading) return;
            await indicatorStore.fetchIndicatorData(indicatorId);
        },

        fetchIndicatorData: async (indicatorId) => {
            const state = get(indicatorStore);
            const indicator = state.availableIndicators.find(i => i.id === indicatorId);
            if (!indicator) return;

            // Set loading state for this indicator
            update(s => ({
                ...s,
                availableIndicators: s.availableIndicators.map(i => 
                    i.id === indicatorId ? { ...i, loading: true, error: null } : i
                )
            }));

            try {
                const marketState = get(marketStore);
                const assetState = get(assetStore);
                
                const symbol = assetState.selectedAsset || 'BTCUSDT';
                const timeframe = marketState.timeframe || '1h';
                const startDateTime = marketState.startDateTime;
                const endDateTime = marketState.endDateTime;

                const queryParams = new URLSearchParams({
                    symbol,
                    interval: timeframe,
                    ...indicator.params
                });

                if (startDateTime) queryParams.set('startDateTime', startDateTime);
                if (endDateTime) queryParams.set('endDateTime', endDateTime);

                const queryKey = `${indicator.route}?${queryParams.toString()}`;

                const res = await fetch(queryKey);
                const data = await res.json();

                if (data.success) {
                    update(s => ({
                        ...s,
                        availableIndicators: s.availableIndicators.map(i => 
                            i.id === indicatorId ? { 
                                ...i, 
                                loading: false, 
                                tvlcData: data.tvlc_data,
                                lastQueryKey: queryKey
                            } : i
                        )
                    }));
                } else {
                    throw new Error(data.error || 'Unknown error');
                }

            } catch (e) {
                console.error(`Error fetching ${indicatorId}:`, e);
                update(s => ({
                    ...s,
                    availableIndicators: s.availableIndicators.map(i => 
                        i.id === indicatorId ? { ...i, loading: false, error: e.message } : i
                    )
                }));
            }
        },

        refreshVisibleIndicators: async () => {
            const state = get(indicatorStore);
            const visibleInds = state.availableIndicators.filter(i => i.visible);
            
            for (const ind of visibleInds) {
                await indicatorStore.fetchIndicatorData(ind.id);
            }
        },

        clearData: () => {
            update(s => ({
                ...s,
                availableIndicators: s.availableIndicators.map(i => ({
                    ...i,
                    data: null,
                    tvlcData: null,
                    error: null
                }))
            }));
        }
    };
}

export const indicatorStore = createIndicatorStore();
