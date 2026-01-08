import { writable } from 'svelte/store';

function createAssetStore() {
    const { subscribe, set, update } = writable({
        assets: [],
        selectedAsset: 'BTCUSDT', // Default
        loading: false,
        error: null
    });

    return {
        subscribe,
        setAssets: (assets) => update(s => ({ ...s, assets })),
        selectAsset: (asset) => update(s => ({ ...s, selectedAsset: asset })),
        setLoading: (loading) => update(s => ({ ...s, loading })),
        setError: (error) => update(s => ({ ...s, error })),
        
        // Actions
        fetchAssets: async () => {
            update(s => ({ ...s, loading: true }));
            try {
                const isDev = import.meta.env.DEV;
                let assets = [];
                
                try {
                    const response = await fetch('/api/config/assets');
                    const text = await response.text();
                    
                    if (!text || text.trim() === '') throw new Error('Empty response');
                    if (text.trim().startsWith('<')) throw new Error('HTML response');
                    
                    const data = JSON.parse(text);
                    if (data.success) {
                        assets = data.assets;
                    } else {
                        throw new Error(data.error);
                    }
                } catch (netError) {
                    if (isDev) {
                        console.warn("Backend unreachable, using mock assets");
                        assets = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'];
                    } else {
                        throw netError;
                    }
                }
                
                update(s => ({ ...s, assets, loading: false }));
            } catch (e) {
                console.error("Failed to fetch assets:", e);
                update(s => ({ ...s, error: e.message, loading: false }));
                // Fallback
                update(s => ({ ...s, assets: ['BTCUSDT', 'ETHUSDT'], loading: false }));
            }
        }
    };
}

export const assetStore = createAssetStore();
