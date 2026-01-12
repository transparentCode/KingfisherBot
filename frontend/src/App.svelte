<script>
  import { onMount } from 'svelte'
  import { authStore } from './stores/authStore.js'
  import { assetStore } from './stores/assetStore.js'
  import { configStore } from './stores/configStore.js'
  import { indicatorStore } from './stores/indicatorStore.js'
  
  import DashboardView from './views/DashboardView.svelte'
  import LogView from './components/LogView.svelte'
  import SystemMetricsView from './components/SystemMetricsView.svelte'
  import Background from './components/Background.svelte';
  import './modal.css'

  let username = ''
  let password = ''
  let showConfigModal = false
  let activeTab = 'dashboard' // dashboard, strategies, backtests, logs

  onMount(() => {
    if ($authStore.token) {
      assetStore.fetchAssets()
      indicatorStore.fetchAvailableIndicators()
    }
  })

  // React to auth changes
  $: if ($authStore.token) {
    assetStore.fetchAssets()
    indicatorStore.fetchAvailableIndicators()
  }

  const handleLogin = async (e) => {
    e.preventDefault()
    await authStore.login(username.trim(), password)
  }

  const handleRefreshConfig = async () => {
    await configStore.reloadConfig()
  }

  const openConfig = async () => {
    await configStore.fetchConfig()
    showConfigModal = true
  }
</script>

{#if showConfigModal}
  <div class="modal-backdrop" role="button" tabindex="0" on:click|self={() => showConfigModal = false} on:keydown={(e) => (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') && (showConfigModal = false)}>
    <div class="modal-card">
      <header>
        <h2>System Configuration</h2>
        <button class="icon-btn" on:click={() => showConfigModal = false}>✕</button>
      </header>
      <div class="modal-body">
        {#if $configStore.loading}
          <p>Loading configuration...</p>
        {:else if $configStore.error}
          <p class="error">{$configStore.error}</p>
        {:else if $configStore.config}
          <pre>{JSON.stringify($configStore.config, null, 2)}</pre>
        {/if}
      </div>
      <footer>
        <button class="secondary" on:click={handleRefreshConfig} disabled={$configStore.loading}>
          {$configStore.loading ? 'Reloading...' : 'Reload from Disk'}
        </button>
        <button class="primary" on:click={() => showConfigModal = false}>Close</button>
      </footer>
    </div>
  </div>
{/if}

{#if !$authStore.token}
  <Background />
  <div class="auth-shell">
    <div class="auth-card">
      <header>
        <p class="eyebrow">KingfisherBot</p>
        <h1>Sign In</h1>
        <p class="muted">Enter your credentials to continue.</p>
      </header>

      <form class="auth-form" on:submit|preventDefault={handleLogin}>
        <label>
          <span>Username</span>
          <input
            type="text"
            bind:value={username}
            autocomplete="username"
            placeholder="admin"
            required
          />
        </label>

        <label>
          <span>Password</span>
          <input
            type="password"
            bind:value={password}
            autocomplete="current-password"
            placeholder="••••••••"
            required
          />
        </label>

        {#if $authStore.error}
          <div class="error-chip">{$authStore.error}</div>
        {/if}

        <button class="primary" type="submit" disabled={$authStore.loading}>
          {$authStore.loading ? 'Signing in…' : 'Sign In'}
        </button>
      </form>
    </div>
  </div>
{:else}
  <Background />
  <div class="app-shell">
    <header class="navbar">
      <div class="brand">
        <div class="dot live"></div>
        <span>KingfisherBot</span>
      </div>

      <nav>
        <button class="nav-btn" class:active={activeTab === 'dashboard'} on:click={() => activeTab = 'dashboard'}>Dashboard</button>
        <button class="nav-btn" class:active={activeTab === 'strategies'} on:click={() => activeTab = 'strategies'}>Strategies</button>
        <button class="nav-btn" class:active={activeTab === 'backtests'} on:click={() => activeTab = 'backtests'}>Backtests</button>
        <button class="nav-btn" class:active={activeTab === 'system'} on:click={() => activeTab = 'system'}>System Metrics</button>
        <button class="nav-btn" class:active={activeTab === 'logs'} on:click={() => activeTab = 'logs'}>System Logs</button>
        <button class="nav-btn" on:click={openConfig}>Settings</button>
      </nav>

      <div class="user-actions">
        <span class="muted">👋 {$authStore.user?.username}</span>
        <button class="ghost" on:click={() => authStore.logout()}>Logout</button>
      </div>
    </header>

    <main class="content">
      {#if activeTab === 'dashboard'}
        <DashboardView />
      {:else if activeTab === 'logs'}
        <LogView />
      {:else if activeTab === 'system'}
        <SystemMetricsView />
      {:else}
        <div class="placeholder-page">
          <h2>{activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}</h2>
          <p>This section is under construction.</p>
        </div>
      {/if}
    </main>
  </div>
{/if}

<style>
  /* Ensure content sits above background */
  .auth-card {
    position: relative;
    z-index: 1;
    background: rgba(30, 30, 30, 0.85); /* Add some transparency/blur */
    backdrop-filter: blur(10px);
  }

  .content {
    position: relative;
    z-index: 1;
  }
</style>
