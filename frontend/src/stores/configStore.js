import { writable } from 'svelte/store';

function createConfigStore() {
    const { subscribe, set, update } = writable({
        config: null,
        loading: false,
        error: null,
        lastUpdated: null
    });

    return {
        subscribe,
        
        fetchConfig: async () => {
            update(s => ({ ...s, loading: true }));
            try {
                // Check if we are in dev mode and potentially without backend
                const isDev = import.meta.env.DEV;
                
                let data;
                try {
                    const response = await fetch('/api/config');
                    const text = await response.text();
                    
                    // Check for empty response
                    if (!text || text.trim() === '') {
                        throw new Error('Empty response from backend');
                    }

                    // Check if response is HTML (proxy failed or backend down)
                    if (text.trim().startsWith('<')) {
                        throw new Error('Backend not available (received HTML)');
                    }
                    
                    data = JSON.parse(text);
                } catch (netError) {
                    if (isDev) {
                        console.warn("Backend unreachable, using mock config data");
                        // Mock data for development
                        data = {
                            success: true,
                            config: {
                                total_assets: 4,
                                enabled_assets: ['BTCUSDT', 'ETHUSDT'],
                                disabled_assets: ['SOLUSDT', 'BNBUSDT'],
                                regime_adaptation_global: true,
                                global_config: {
                                    regime_adaptation: { enabled: true, update_interval: '15m' },
                                    default_timeframes: ['15m', '1h', '4h']
                                },
                                asset_details: {
                                    'BTCUSDT': { enabled: true, regime_enabled: true },
                                    'ETHUSDT': { enabled: true, regime_enabled: false }
                                }
                            }
                        };
                    } else {
                        throw netError;
                    }
                }
                
                if (data.success) {
                    update(s => ({ 
                        ...s, 
                        config: data.config, 
                        loading: false,
                        lastUpdated: new Date()
                    }));
                } else {
                    throw new Error(data.error);
                }
            } catch (e) {
                console.error("Failed to fetch config:", e);
                update(s => ({ ...s, error: e.message, loading: false }));
            }
        },

        reloadConfig: async () => {
            update(s => ({ ...s, loading: true }));
            try {
                const isDev = import.meta.env.DEV;
                let data;

                try {
                    const response = await fetch('/api/config/reload', { method: 'POST' });
                    const text = await response.text();

                    if (!text || text.trim() === '') throw new Error('Empty response');
                    if (text.trim().startsWith('<')) throw new Error('HTML response');

                    data = JSON.parse(text);
                } catch (netError) {
                    if (isDev) {
                        console.warn("Backend unreachable, simulating config reload");
                        // Simulate network delay
                        await new Promise(r => setTimeout(r, 800));
                        data = { success: true, message: "Mock reload successful" };
                    } else {
                        throw netError;
                    }
                }
                
                if (data.success) {
                    // Re-fetch to get new values (fetchConfig handles its own mocking)
                    // We need to call the internal fetchConfig logic or just the exported one.
                    // Since we are inside the factory, we can't easily call the exported 'configStore.fetchConfig()' 
                    // but we can replicate the call or structure this better. 
                    // Actually, we can just return true and let the component call fetchConfig if needed, 
                    // OR we can implement the re-fetch logic here.
                    
                    // Let's re-use the fetch logic by calling the internal fetchConfig if we could, 
                    // but since it's defined in the return object, we can't access it directly easily 
                    // without defining it outside.
                    // Instead, I'll just manually trigger the fetch logic again or rely on the component.
                    // But the original code did a fetch. Let's do that.
                    
                    // We'll just call the fetch API again, and let the same fallback logic handle it.
                    // But wait, I can't call 'fetchConfig' from here easily because it's a sibling property.
                    // I will just duplicate the fetch call logic or better yet, 
                    // since I can't easily call the sibling method, I will just update the state 
                    // if it was a mock reload, or do a real fetch if it was a real reload.
                    
                    if (isDev && !data.config) {
                         // If it was a mock reload, we might want to update the timestamp
                         update(s => ({ ...s, loading: false, lastUpdated: new Date() }));
                    } else {
                         // Real reload, fetch updated config
                         // We can use the same robust fetch logic as fetchConfig
                         try {
                             const configResponse = await fetch('/api/config');
                             const configText = await configResponse.text();
                             if (!configText.trim().startsWith('<')) {
                                 const configData = JSON.parse(configText);
                                 if (configData.success) {
                                     update(s => ({ 
                                         ...s, 
                                         config: configData.config, 
                                         loading: false, 
                                         lastUpdated: new Date()
                                     }));
                                 }
                             }
                         } catch (e) {
                             // If re-fetch fails in dev, just stop loading
                             if (isDev) update(s => ({ ...s, loading: false, lastUpdated: new Date() }));
                         }
                    }
                    return true;
                } else {
                    throw new Error(data.error);
                }
            } catch (e) {
                console.error("Failed to reload config:", e);
                update(s => ({ ...s, error: e.message, loading: false }));
                return false;
            }
        }
    };
}

export const configStore = createConfigStore();
