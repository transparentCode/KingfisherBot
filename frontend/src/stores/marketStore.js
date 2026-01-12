import { writable } from 'svelte/store';

function createMarketStore() {
    // Abort controllers for network optimization
    let regimeController = null;
    let candlesController = null;

    const { subscribe, set, update } = writable({
        symbol: 'BTCUSDT',
        timeframe: '1h',
        candles: [],
        tvlcData: { candles: [], lines: [], markers: [], histograms: [] },
        regime: null,
        loading: false,
        regimeLoading: false,
        error: null,
        lastUpdated: null,
        startDateTime: null,
        endDateTime: null
    });

    return {
        subscribe,
        
        fetchRegime: async (symbol, timeframe) => {
            if (regimeController) regimeController.abort();
            regimeController = new AbortController();

            update(s => ({ ...s, regimeLoading: true }));
            try {
                const response = await fetch(`/api/asset/regime/${symbol}?timeframe=${timeframe}`, {
                    signal: regimeController.signal
                });
                const data = await response.json();
                if (data.success) {
                    update(s => ({ ...s, regime: data.data, regimeLoading: false }));
                } else {
                    console.warn("Regime fetch failed:", data.error);
                    update(s => ({ ...s, regime: null, regimeLoading: false }));
                }
            } catch (e) {
                if (e.name === 'AbortError') return; // Ignore aborted requests
                console.error("Failed to fetch regime:", e);
                update(s => ({ ...s, regime: null, regimeLoading: false }));
            }
        },

        fetchCandles: async (symbol, timeframe = '1h', startDateTime, endDateTime) => {
            // Cancel previous pending request
            if (candlesController) candlesController.abort();
            candlesController = new AbortController();

            // Default range: last 30 days ending now
            const end = endDateTime ? new Date(endDateTime) : new Date();
            const start = startDateTime ? new Date(startDateTime) : new Date(end.getTime() - 30 * 24 * 60 * 60 * 1000);

            const startISO = start.toISOString();
            const endISO = end.toISOString();

            update(s => ({ ...s, loading: true, symbol, timeframe, startDateTime: startISO, endDateTime: endISO }));
            try {
                const isDev = import.meta.env.DEV;
                let candles = [];
                let tvlcData = { candles: [], lines: [], markers: [], histograms: [] };

                try {
                    const params = new URLSearchParams({
                        timeframe,
                        startDateTime: startISO,
                        endDateTime: endISO
                    });

                    const response = await fetch(`/api/asset/candle-data/${symbol}?${params.toString()}`, {
                        signal: candlesController.signal
                    });
                    const text = await response.text();

                    if (!text || text.trim() === '') throw new Error('Empty response');
                    if (text.trim().startsWith('<')) throw new Error('HTML response');

                    const data = JSON.parse(text);
                    if (data.success) {
                        candles = data.data.candles;
                        if (data.data.tvlc_data) {
                            tvlcData = {
                                candles: data.data.tvlc_data.candles || [],
                                lines: data.data.tvlc_data.lines || [],
                                markers: data.data.tvlc_data.markers || [],
                                histograms: data.data.tvlc_data.histograms || []
                            };
                        }
                    } else {
                        throw new Error(data.error);
                    }
                } catch (netError) {
                    if (netError.name === 'AbortError') throw netError; // Re-throw cancel
                    
                    if (isDev) {
                        console.warn(`Backend unreachable, using mock candles for ${symbol}`);
                        // Generate mock candles
                        const now = Math.floor(Date.now() / 1000);
                        let price = 40000 + Math.random() * 1000;
                        for (let i = 0; i < 100; i++) {
                            const time = now - (100 - i) * 3600;
                            const open = price;
                            const close = price + (Math.random() - 0.5) * 200;
                            const high = Math.max(open, close) + Math.random() * 50;
                            const low = Math.min(open, close) - Math.random() * 50;
                            const volume = Math.random() * 100;
                            
                            candles.push({ time, open, high, low, close, volume });
                            price = close;
                        }

                        tvlcData = {
                            candles: candles.map(c => ({
                                time: c.time,
                                open: c.open,
                                high: c.high,
                                low: c.low,
                                close: c.close,
                                volume: c.volume
                            })),
                            lines: [],
                            markers: [],
                            histograms: []
                        };
                    } else {
                        throw netError;
                    }
                }

                update(s => ({ 
                    ...s,
                    candles,
                    tvlcData,
                    loading: false,
                    error: null,
                    lastUpdated: new Date()
                }));
            } catch (e) {
                if (e.name === 'AbortError') return; // Ignore aborted requests
                console.error("Failed to fetch candles:", e);
                update(s => ({ ...s, error: e.message, loading: false }));
            }
        }
    };
}

export const marketStore = createMarketStore();
export const activeIndicators = writable([]);

// Placeholder for WebSocket logic
export function connectWebSocket(symbol) {
    console.log(`Connecting to WebSocket for ${symbol}...`);
    // Implementation will go here
}
