const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('api', {
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  startScan: (p) => ipcRenderer.invoke('start-scan', p),
  onProgress: (cb) => ipcRenderer.on('scan-progress', (e, data) => cb(data)),
  onDone: (cb) => ipcRenderer.on('scan-done', (e, data) => cb(data)),
  onMusicscanStderr: (cb) => ipcRenderer.on('musicscan-stderr', (e, data) => cb(data)),
  onMusicscanStdoutLine: (cb) => ipcRenderer.on('musicscan-stdout-line', (e, data) => cb(data)),
  onMusicscanProgress: (cb) => ipcRenderer.on('musicscan-progress', (e, data) => cb(data)),
  onMusicscanComplete: (cb) => ipcRenderer.on('musicscan-complete', (e, data) => cb(data))
  ,
  /* Party control API */
  startParty: (opts) => ipcRenderer.invoke('start-party', opts),
  setEnergyTarget: (val) => ipcRenderer.invoke('set-energy-target', val),
  adjustFamiliarityBias: (delta) => ipcRenderer.invoke('adjust-familiarity-bias', delta),
  recordPlay: (params) => ipcRenderer.invoke('record-play', params),
  playbackPlay: (params) => ipcRenderer.invoke('playback-play', params),
  playbackPause: (params) => ipcRenderer.invoke('playback-pause', params),
  playbackSkip: (params) => ipcRenderer.invoke('playback-skip', params),
  selectInitial: () => ipcRenderer.invoke('select-initial'),
  hasScanCache: () => ipcRenderer.invoke('has-scan-cache'),
  getLastScan: () => ipcRenderer.invoke('get-last-scan'),
  getFeatures: (p) => ipcRenderer.invoke('get-features', p),
  checkFolderCache: (p) => ipcRenderer.invoke('check-folder-cache', p),
  pauseScan: () => ipcRenderer.invoke('pause-scan'),
  resumeScan: () => ipcRenderer.invoke('resume-scan'),
  onPartyState: (cb) => ipcRenderer.on('party-state', (e, data) => cb(data)),
  onPartyTrack: (cb) => ipcRenderer.on('party-track', (e, data) => cb(data))
  ,
  onPartySkip: (cb) => ipcRenderer.on('party-skip', (e, data) => cb(data))
  ,
  // Guest events
  onGuestUrl: (cb) => ipcRenderer.on('party-guest', (e, data) => cb(data)),
  onGuestRequests: (cb) => ipcRenderer.on('party-guest-requests', (e, data) => cb(data)),
  injectRequest: (id) => ipcRenderer.invoke('inject-request', id),
  onGuestInject: (cb) => ipcRenderer.on('party-guest-inject', (e, data) => cb(data))
  ,
  // Advanced & feedback APIs
  setAdvanced: (opts) => ipcRenderer.invoke('set-advanced', opts),
  setPrompt: (text) => ipcRenderer.invoke('set-prompt', text),
  onPartyFeedback: (cb) => ipcRenderer.on('party-feedback', (e, data) => cb(data)),
  onPartyAsk: (cb) => ipcRenderer.on('party-ask', (e, data) => cb(data)),
  failureResponse: (choice) => ipcRenderer.invoke('failure-response', choice)
})
