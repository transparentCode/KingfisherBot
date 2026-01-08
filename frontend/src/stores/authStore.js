import { writable } from 'svelte/store';

// Basic in-memory auth state. Token is persisted in localStorage.
const storedToken = typeof localStorage !== 'undefined' ? localStorage.getItem('kf_auth_token') : null;
const storedUser = typeof localStorage !== 'undefined' ? localStorage.getItem('kf_auth_user') : null;

const initialState = {
  user: storedToken ? { username: storedUser || 'User' } : null,
  token: storedToken,
  loading: false,
  error: null
};

const createAuthStore = () => {
  const { subscribe, update, set } = writable(initialState);

  return {
    subscribe,
    async login(username, password) {
      update((state) => ({ ...state, loading: true, error: null }));
      try {
        const res = await fetch('/api/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password })
        });

        if (!res.ok) {
          throw new Error('Invalid credentials');
        }

        const data = await res.json();
        const token = data?.token || null;
        if (token && typeof localStorage !== 'undefined') {
          localStorage.setItem('kf_auth_token', token);
          localStorage.setItem('kf_auth_user', username);
        }

        set({ user: { username }, token, loading: false, error: null });
      } catch (err) {
        set({ user: null, token: null, loading: false, error: err.message || 'Login failed' });
      }
    },
    logout() {
      if (typeof localStorage !== 'undefined') {
        localStorage.removeItem('kf_auth_token');
        localStorage.removeItem('kf_auth_user');
      }
      set({ user: null, token: null, loading: false, error: null });
    }
  };
};

export const authStore = createAuthStore();
