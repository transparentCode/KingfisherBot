<script>
  import { onMount, onDestroy } from 'svelte';
  import { io } from 'socket.io-client';

  let logs = [];
  let socket;
  let visible = false;
  let autoScroll = true;
  let logContainer;

  onMount(() => {
    // Connect to the same host as the frontend (proxy handles /socket.io)
    socket = io('/', {
      transports: ['websocket', 'polling']
    });

    socket.on('connect', () => {
      console.log('LogViewer connected to WebSocket');
    });

    socket.on('log_message', (log) => {
      logs = [...logs, log];
      if (logs.length > 1000) {
        logs = logs.slice(-1000);
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

  function toggleVisibility() {
    visible = !visible;
    if (visible) scrollToBottom();
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
</script>

<div class="log-viewer-container" class:visible>
  <div class="header" on:click={toggleVisibility}>
    <span class="title">System Logs</span>
    <div class="controls">
      <span class="count">{logs.length}</span>
      <button on:click|stopPropagation={clearLogs} title="Clear Logs">🚫</button>
      <button on:click|stopPropagation={() => autoScroll = !autoScroll} class:active={autoScroll} title="Auto-scroll">⬇</button>
      <span class="toggle-icon">{visible ? '▼' : '▲'}</span>
    </div>
  </div>

  {#if visible}
    <div class="log-body" bind:this={logContainer}>
      {#each logs as log}
        <div class="log-entry">
          <span class="time">{log.timestamp.split('T')[1].slice(0, 12)}</span>
          <span class="level" style="color: {getLevelColor(log.level)}">{log.level}</span>
          <span class="module">[{log.name}]</span>
          <span class="message">{log.message}</span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .log-viewer-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #1e1e1e;
    border-top: 1px solid #333;
    z-index: 1000;
    font-family: monospace;
    font-size: 12px;
    transition: height 0.3s;
  }

  .header {
    padding: 5px 10px;
    background: #252526;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
    user-select: none;
  }

  .title {
    font-weight: bold;
    color: #ccc;
  }

  .controls {
    display: flex;
    gap: 10px;
    align-items: center;
  }

  .controls button {
    background: none;
    border: none;
    color: #888;
    cursor: pointer;
    padding: 0;
    font-size: 14px;
  }

  .controls button:hover {
    color: #fff;
  }

  .controls button.active {
    color: #4caf50;
  }

  .log-body {
    height: 200px;
    overflow-y: auto;
    padding: 5px;
    background: #1e1e1e;
    color: #d4d4d4;
  }

  .log-entry {
    padding: 2px 0;
    border-bottom: 1px solid #2a2a2a;
    white-space: pre-wrap;
    word-break: break-all;
  }

  .time {
    color: #569cd6;
    margin-right: 5px;
  }

  .level {
    font-weight: bold;
    margin-right: 5px;
    min-width: 50px;
    display: inline-block;
  }

  .module {
    color: #9cdcfe;
    margin-right: 5px;
  }

  .message {
    color: #d4d4d4;
  }
</style>