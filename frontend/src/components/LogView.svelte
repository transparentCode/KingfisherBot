<script>
  import { onMount, onDestroy } from 'svelte';
  import { io } from 'socket.io-client';

  let logs = [];
  let socket;
  let autoScroll = true;
  let logContainer;
  let filterLevel = 'ALL'; // ALL, INFO, WARNING, ERROR

  onMount(() => {
    socket = io('/', {
      transports: ['websocket', 'polling']
    });

    socket.on('connect', () => {
      console.log('LogView connected to WebSocket');
    });

    socket.on('connect_error', (err) => {
      console.error('LogView WebSocket Connection Error:', err);
    });

    socket.on('log_message', (log) => {
      logs = [...logs, log];
      if (logs.length > 2000) {
        logs = logs.slice(-2000);
      }
      if (autoScroll) {
        scrollToBottom();
      }
    });

    return () => {
      if (socket) socket.disconnect();
    };
  });

  function scrollToBottom() {
    if (logContainer) {
      setTimeout(() => {
        logContainer.scrollTop = logContainer.scrollHeight;
      }, 0);
    }
  }

  function clearLogs() {
    logs = [];
  }

  function getLevelColor(level) {
    switch (level) {
      case 'INFO': return '#4caf50';
      case 'WARNING': return '#ff9800';
      case 'ERROR': return '#f44336';
      case 'CRITICAL': return '#d32f2f';
      case 'DEBUG': return '#2196f3';
      default: return '#b0bec5';
    }
  }

  $: filteredLogs = filterLevel === 'ALL' ? logs : logs.filter(l => l.level === filterLevel);
</script>

<div class="log-view-page">
  <header class="log-header">
    <div class="left">
      <h2>System Logs</h2>
      <div class="filters">
        <button class:active={filterLevel === 'ALL'} on:click={() => filterLevel = 'ALL'}>All</button>
        <button class:active={filterLevel === 'INFO'} on:click={() => filterLevel = 'INFO'}>Info</button>
        <button class:active={filterLevel === 'WARNING'} on:click={() => filterLevel = 'WARNING'}>Warning</button>
        <button class:active={filterLevel === 'ERROR'} on:click={() => filterLevel = 'ERROR'}>Error</button>
      </div>
    </div>
    <div class="right">
      <span class="count">{filteredLogs.length} events</span>
      <button class="icon-btn" on:click={clearLogs} title="Clear">🚫</button>
      <button class="icon-btn" class:active={autoScroll} on:click={() => autoScroll = !autoScroll} title="Auto-scroll">⬇</button>
    </div>
  </header>

  <div class="console-container" bind:this={logContainer}>
    {#each filteredLogs as log}
      <div class="log-line">
        <span class="timestamp">{log.timestamp}</span>
        <span class="level" style="color: {getLevelColor(log.level)}">{log.level}</span>
        <span class="source">[{log.name}]</span>
        <span class="message">{log.message}</span>
      </div>
    {/each}
    {#if filteredLogs.length === 0}
      <div class="empty-state">Waiting for logs...</div>
    {/if}
  </div>
</div>

<style>
  .log-view-page {
    display: flex;
    flex-direction: column;
    height: 100%;
    background: #1e1e1e;
    color: #d4d4d4;
    font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
  }

  .log-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 20px;
    background: #252526;
    border-bottom: 1px solid #333;
  }

  .log-header h2 {
    margin: 0;
    font-size: 1.2rem;
    color: #ccc;
    margin-right: 20px;
  }

  .left, .right {
    display: flex;
    align-items: center;
    gap: 15px;
  }

  .filters button {
    background: none;
    border: 1px solid transparent;
    color: #888;
    padding: 4px 8px;
    cursor: pointer;
    border-radius: 4px;
    font-size: 0.85rem;
  }

  .filters button:hover {
    color: #fff;
    background: rgba(255, 255, 255, 0.05);
  }

  .filters button.active {
    background: rgba(255, 255, 255, 0.1);
    color: #fff;
    border-color: rgba(255, 255, 255, 0.2);
  }

  .icon-btn {
    background: none;
    border: none;
    color: #888;
    cursor: pointer;
    font-size: 1.1rem;
    padding: 5px;
    border-radius: 4px;
  }

  .icon-btn:hover {
    background: rgba(255, 255, 255, 0.1);
    color: #fff;
  }

  .icon-btn.active {
    color: #4caf50;
  }

  .console-container {
    flex: 1;
    overflow-y: auto;
    padding: 10px;
    background: #1e1e1e;
  }

  .log-line {
    padding: 2px 0;
    border-bottom: 1px solid #2a2a2a;
    white-space: pre-wrap;
    word-break: break-all;
    font-size: 13px;
    line-height: 1.4;
  }

  .log-line:hover {
    background: #2a2d2e;
  }

  .timestamp {
    color: #569cd6;
    margin-right: 10px;
    opacity: 0.8;
  }

  .level {
    font-weight: bold;
    margin-right: 10px;
    min-width: 60px;
    display: inline-block;
  }

  .source {
    color: #9cdcfe;
    margin-right: 10px;
  }

  .message {
    color: #d4d4d4;
  }

  .empty-state {
    padding: 20px;
    text-align: center;
    color: #666;
    font-style: italic;
  }
</style>