<script>
  import { onMount } from 'svelte';
  import { indicatorStore } from '../stores/indicatorStore';

  export let close;

  onMount(() => {
    if ($indicatorStore.availableCategories.length === 0) {
      indicatorStore.fetchAvailableIndicators();
    }
  });

  const handleAdd = (indicatorId) => {
    indicatorStore.addIndicator(indicatorId);
    // We don't close the modal so user can add multiple
  };
</script>

<div class="modal-backdrop" on:click|self={close}>
  <div class="modal-card indicator-modal">
    <header>
      <h2>Add Indicators</h2>
      <button class="icon-btn" on:click={close}>✕</button>
    </header>
    
    <div class="modal-body">
      {#if $indicatorStore.loading && $indicatorStore.availableCategories.length === 0}
        <div class="loading-state">Loading indicators...</div>
      {:else if $indicatorStore.error}
        <div class="error-state">{$indicatorStore.error}</div>
      {:else}
        <div class="indicator-list">
          {#each $indicatorStore.availableCategories as category}
            <div class="category-section">
              <h3>{category.category}</h3>
              <div class="grid-list">
                {#each category.indicators as indicator}
                  <button class="indicator-item" on:click={() => handleAdd(indicator.indicator_id)}>
                    <div class="indicator-info">
                      <span class="name">{indicator.display_name}</span>
                      <span class="desc">{indicator.description}</span>
                    </div>
                    <span class="add-icon">+</span>
                  </button>
                {/each}
              </div>
            </div>
          {/each}
        </div>
      {/if}
    </div>
    
    <footer>
      <button class="primary" on:click={close}>Done</button>
    </footer>
  </div>
</div>

<style>
  .indicator-modal {
    max-width: 600px;
    height: 80vh;
  }

  .category-section {
    margin-bottom: 24px;
  }

  .category-section h3 {
    font-size: 0.9rem;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
  }

  .grid-list {
    display: grid;
    grid-template-columns: 1fr;
    gap: 8px;
  }

  .indicator-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--panel-2);
    border: 1px solid var(--border);
    padding: 12px;
    border-radius: 8px;
    cursor: pointer;
    text-align: left;
    transition: all 0.2s;
    color: var(--text);
  }

  .indicator-item:hover {
    border-color: var(--accent);
    background: var(--bg);
  }

  .indicator-info {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }

  .name {
    font-weight: 600;
    font-size: 0.95rem;
  }

  .desc {
    font-size: 0.8rem;
    color: var(--muted);
  }

  .add-icon {
    font-size: 1.2rem;
    color: var(--accent);
    font-weight: bold;
  }
</style>
