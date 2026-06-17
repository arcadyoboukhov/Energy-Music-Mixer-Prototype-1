// Minimal UI scripting for onboarding and party controls
const onboardEl = document.getElementById('onboard')
const mainUIEl = document.getElementById('mainUI')
const vibeButtons = document.querySelectorAll('.vibe')
const startPartyBtn = document.getElementById('startParty')
const energySlider = document.getElementById('energySlider')
const sliderValue = document.getElementById('sliderValue')
const energyValue = document.getElementById('energyValue')
const energyTrend = document.getElementById('energyTrend')
const moreHypeBtn = document.getElementById('moreHype')
const chillDownBtn = document.getElementById('chillDown')
// singalong UI removed
const currentVibe = document.getElementById('currentVibe')
const partyStatus = document.getElementById('partyStatus')
const nowTrack = document.getElementById('nowTrack')
const nextTrack = document.getElementById('nextTrack')
const guestUrlEl = document.getElementById('guestUrl')
const guestQREl = document.getElementById('guestQR')
const guestListEl = document.getElementById('guestList')
const feedbackMessageEl = document.getElementById('feedbackMessage')
const advancedToggle = document.getElementById('advancedToggle')
const advancedPanel = document.getElementById('advancedPanel')
const genreFilterEl = document.getElementById('genreFilter')
const explicitToggleEl = document.getElementById('explicitToggle')
const tempoBiasEl = document.getElementById('tempoBias')
const tempoBiasValueEl = document.getElementById('tempoBiasValue')
const saveAdvancedBtn = document.getElementById('saveAdvanced')
const playBtn = document.getElementById('playBtn')
const pauseBtn = document.getElementById('pauseBtn')
const skipBtn = document.getElementById('skipBtn')
const failurePromptEl = document.getElementById('failurePrompt')
const failurePromptText = document.getElementById('failurePromptText')
const failureKeepBtn = document.getElementById('failureKeep')
const failureSwitchBtn = document.getElementById('failureSwitch')
const importFilesBtn = document.getElementById('importFiles')
    try {
      if (best) {
        nowTrack.textContent = best
        // next track: pick second-best if available
        const next = (Array.isArray(obj.scored) && obj.scored.length > 1) ? obj.scored[1].path : '(queued)'
        nextTrack.textContent = next

        // if currently playing, switch audio to new track
        try {
          if (best) {
            await playPath(best)
          }
        } catch (e) { console.warn('onPartyTrack playback failed', e) }

        // notify playback/memory: record play-start (played_at in seconds)
        try { await window.api.recordPlay({ path: best, played_at: Date.now() / 1000.0, duration: null }) } catch (e) { console.warn('recordPlay failed', e) }
      }
let audioA = new Audio()
let audioB = new Audio()
audioA.preload = 'auto'; audioB.preload = 'auto'
audioA.crossOrigin = 'anonymous'; audioB.crossOrigin = 'anonymous'
let srcA = null; let srcB = null
try {
  srcA = audioCtx.createMediaElementSource(audioA)
  srcB = audioCtx.createMediaElementSource(audioB)
} catch (e) {
  // some environments may not allow MediaElementSource until user gesture
}
let gainA = audioCtx.createGain(); let gainB = audioCtx.createGain();
try { srcA.connect(gainA); gainA.connect(audioCtx.destination); srcB.connect(gainB); gainB.connect(audioCtx.destination) } catch (e) {}
gainA.gain.value = 1.0; gainB.gain.value = 0.0
let activeEl = audioA; let inactiveEl = audioB
let activeGain = gainA; let inactiveGain = gainB
let isPlaying = false
// singalong removed: no singalong state

// last-scanned folder will be populated after DOMContentLoaded (preload may not be ready yet)

function clamp(v, a=0, b=100){ return Math.max(a, Math.min(b, v)) }
// safe subscriber helper: avoids calling missing API functions
function safeOnAPI(name, cb) {
  try {
    if (window.api && typeof window.api[name] === 'function') {
      window.api[name](cb)
    } else {
      console.warn('API listener not available:', name)
    }
  } catch (e) {
    console.warn('safeOnAPI error for', name, e)
  }
}

// vibe selection
vibeButtons.forEach(b => {
  b.addEventListener('click', () => {
    vibeButtons.forEach(x => x.style.borderColor = 'rgba(255,255,255,0.06)')
    b.style.borderColor = '#6ad3a2'
    selectedVibe = b.dataset.vibe
  })
})

// (Spotify integration removed)

if (importFilesBtn) {
// Robust import handler: use a function so we can delegate clicks even if
// the captured DOM reference is missing. Shows UI feedback and recovers on
// cancel/failure.
async function handleImportClick() {
  const importBtn = document.getElementById('importFiles')
  const statusEl = document.getElementById('scanStatus')
  try {
    if (importBtn) importBtn.disabled = true
    if (statusEl) statusEl.textContent = 'Opening folder chooser...'

    // prefer native folder chooser via main process
    if (window.api && typeof window.api.selectFolder === 'function') {
      const p = await window.api.selectFolder()
      if (!p) {
        if (importBtn) importBtn.disabled = false
        if (statusEl) statusEl.textContent = 'Import cancelled'
        return
      }
      selectedFolder = p
    } else if (window.api && typeof window.api.startScan === 'function') {
      // fallback: use a hidden <input webkitdirectory> to let user pick a folder
      const input = document.createElement('input')
      input.type = 'file'
      input.webkitdirectory = true
      input.multiple = true
      input.style.display = 'none'
      document.body.appendChild(input)
      const picked = await new Promise((resolve) => {
        input.addEventListener('change', () => resolve(Array.from(input.files || [])), { once: true })
        input.click()
      })
      try { document.body.removeChild(input) } catch (e) {}
      if (!picked || picked.length === 0) {
        if (importBtn) importBtn.disabled = false
        if (statusEl) statusEl.textContent = 'Import cancelled'
        return
      }
      // derive folder from first file path
      const first = picked[0]
      const fp = first.path || first.webkitRelativePath || first.name
      const idx = Math.max(fp.lastIndexOf('\\'), fp.lastIndexOf('/'))
      const folder = idx > 0 ? fp.slice(0, idx) : fp
      selectedFolder = folder
    } else {
      if (importBtn) importBtn.disabled = false
      if (statusEl) statusEl.textContent = 'Import not available'
      alert('Import not available (internal API missing). Try restarting the app.')
      return
    }

    // start-scan UI (update immediately)
    if (importBtn) importBtn.textContent = 'Checking cache…'
    startPartyBtn.disabled = true
    scanTotal = 0; scannedCount = 0; scanning = true; scanComplete = false
    scannedPaths = new Set()
    const progressEl = document.getElementById('scanProgress')
    if (progressEl) progressEl.value = 0
    if (statusEl) statusEl.textContent = 'Checking cache…'

    // quick cache check before starting a full scan so the UI reflects
    // cached state immediately (main will also run its own checker).
    let cacheChecked = false
    try {
      if (window.api && typeof window.api.checkFolderCache === 'function') {
        const chk = await window.api.checkFolderCache(selectedFolder)
        if (chk && chk.all_cached) {
          const cnt = chk.count || 0
          scanTotal = cnt
          scannedCount = cnt
          if (progressEl) progressEl.value = 100
          if (importBtn) { importBtn.textContent = 'Imported'; importBtn.disabled = false }
          if (statusEl) statusEl.textContent = `All ${cnt} tracks already cached`
          startPartyBtn.disabled = false
          scanComplete = true
          scanning = false
          cacheChecked = true
        } else if (chk && typeof chk.count === 'number') {
          // show the quick counts while a full scan proceeds
          scanTotal = chk.count
          if (statusEl) statusEl.textContent = `Found ${chk.count} tracks — starting scan…`
        }
      }
    } catch (e) {
      console.warn('cache check failed', e)
    }

    if (!cacheChecked) {
      if (importBtn) importBtn.textContent = 'Scanning…'
      if (statusEl) statusEl.textContent = 'Counting files…'
      // invoke main scan
      try {
        const ok = await window.api.startScan(selectedFolder)
        if (!ok) {
          if (importBtn) importBtn.textContent = 'Import (failed)'
          if (statusEl) statusEl.textContent = 'Scan start failed'
          scanning = false
          if (importBtn) importBtn.disabled = false
        }
      } catch (e) {
        if (importBtn) importBtn.textContent = 'Import (failed)'
        if (statusEl) statusEl.textContent = 'Scan start failed'
        scanning = false
        console.error('startScan failed', e)
        if (importBtn) importBtn.disabled = false
      }
    }
  } catch (e) {
    console.error('importFiles handler failed', e)
    if (importBtn) importBtn.disabled = false
    if (statusEl) statusEl.textContent = 'Import failed: ' + String(e)
    alert('Import failed: ' + String(e))
  }
}

// expose handler globally so DevTools and fallback listeners can call it
try { window.handleImportClick = handleImportClick } catch (e) {}

// Also bind the button directly so clicks always trigger the handler and
// provide immediate visual feedback even if delegation fails.
try {
    if (importFilesBtn) {
    importFilesBtn.addEventListener('click', (ev) => {
      try {
        ev.preventDefault()
        // prevent the document-level delegated click handler from also
        // invoking the import handler (causes duplicate folder chooser)
        try { ev.stopPropagation() } catch (e) {}
        const statusEl = document.getElementById('scanStatus')
        if (statusEl) statusEl.textContent = 'Import clicked…'
        importFilesBtn.disabled = true
        // fire and forget; errors are handled inside
        try { (window.handleImportClick || handleImportClick)() } catch (e) { console.warn('import click invoke failed', e) }
      } catch (e) {
        console.error('importFiles direct click failed', e)
        const statusEl = document.getElementById('scanStatus')
        if (statusEl) statusEl.textContent = 'Import failed: ' + String(e)
        if (importFilesBtn) importFilesBtn.disabled = false
      }
    })
  }
} catch (e) { console.warn('failed to bind importFilesBtn click', e) }

// Delegate clicks to ensure the handler runs even if a prior script error
// prevented element capture. This also handles clicks on child elements.
document.addEventListener('click', (ev) => {
  try {
    const t = ev.target
    if (!t) return
    const btn = t.closest && t.closest('#importFiles')
    if (btn || (t.id && t.id === 'importFiles')) {
      try { (window.handleImportClick || handleImportClick)() } catch (e) { console.warn('delegate invoke failed', e) }
    }
  } catch (e) {
    console.error('click delegation error', e)
  }
})
} else {
  console.warn('Import button not found in DOM')
}

// show musicscan stderr messages as user-visible status updates
document.addEventListener('DOMContentLoaded', () => {
  try {
    if (window.api && typeof window.api.onMusicscanStderr === 'function') {
      window.api.onMusicscanStderr((msg) => {
        try {
          const statusEl = document.getElementById('scanStatus')
          const text = (msg || '').toString()
          // treat check_musicscan_cache progress lines as status updates
          if (text.startsWith('check_musicscan_cache:')) {
            const short = text.replace(/^check_musicscan_cache:\s*/i, '')
            if (statusEl) statusEl.textContent = short
            return
          }
          // otherwise treat as an error
          if (statusEl) statusEl.textContent = 'Scan error: ' + text
          if (importFilesBtn) { importFilesBtn.textContent = 'Import (error)'; importFilesBtn.disabled = false }
          scanning = false
        } catch (e) { console.warn('onMusicscanStderr handler error', e) }
      })
    }
  } catch (e) {}
})

startPartyBtn.addEventListener('click', async () => {
  if (!scanComplete) {
    alert('Please select a music folder and allow the scan to finish before starting the party.')
    return
  }

  await window.api.startParty({ vibe: selectedVibe })
  // request initial selection from server (uses cached scan results)
  try {
    const res = await window.api.selectInitial()
    if (res && res.best) {
      nowTrack.textContent = res.best
      const next = (Array.isArray(res.scored) && res.scored.length > 1) ? res.scored[1].path : '(queued)'
      nextTrack.textContent = next
      // autoplay selected initial track
      try {
        try { await playPath(res.best) } catch (e) { console.warn('autoplay failed', e) }
        try { await window.api.recordPlay({ path: res.best, played_at: Date.now() / 1000.0 }) } catch (e) {}
      } catch (e) { console.warn('autoplay failed', e) }
    }
  } catch (e) { console.warn('selectInitial failed', e) }

  // switch to main UI
  onboardEl.classList.add('hidden')
  mainUIEl.classList.remove('hidden')
  currentVibe.textContent = selectedVibe || 'Party'
})

// slider debounce
let sliderTimer = null
energySlider.addEventListener('input', (e) => {
  const v = Number(e.target.value)
  sliderValue.textContent = `(${v})`
  if (sliderTimer) clearTimeout(sliderTimer)
  sliderTimer = setTimeout(() => {
    window.api.setEnergyTarget(v)
  }, 120)
})

function setEnergyTargetRelative(delta){
  const target = lastParty && lastParty.target !== undefined ? lastParty.target : Number(energySlider.value || 65)
  const next = clamp(target + delta)
  energySlider.value = next
  sliderValue.textContent = `(${next})`
  window.api.setEnergyTarget(next)
}

moreHypeBtn.addEventListener('click', () => setEnergyTargetRelative(15))
chillDownBtn.addEventListener('click', () => setEnergyTargetRelative(-15))
// helper: ensure audio context resumed on user gesture
async function _resumeAudioCtx(){ try { if (audioCtx && audioCtx.state !== 'running') await audioCtx.resume() } catch(e){} }

async function playImmediate(url, startTime = 0){
  if (!url) return
  await _resumeAudioCtx()
  try {
    activeEl.src = url
    activeGain.gain.cancelScheduledValues(audioCtx.currentTime)
    activeGain.gain.setValueAtTime(1.0, audioCtx.currentTime)
    // play at currentTime (may be set by caller via loadedmetadata)
    const startPlay = async () => {
      try {
        try {
          if (startTime && typeof activeEl.currentTime !== 'undefined') {
            try { activeEl.currentTime = startTime } catch (__) {}
          }
        } catch (__) {}
        await activeEl.play()
      } catch (e) {
        try { activeEl.play() } catch (__) {}
      }
    }
    if (activeEl.readyState >= 1) {
      await startPlay()
    } else {
      await new Promise((resolve) => {
        activeEl.addEventListener('loadedmetadata', async () => { await startPlay(); resolve() }, { once: true })
      })
    }
    isPlaying = true
    partyStatus.textContent = 'Playing'
  } catch (e) { console.warn('playImmediate failed', e) }
}

function _swapActive(){
  const tEl = activeEl; activeEl = inactiveEl; inactiveEl = tEl
  const tG = activeGain; activeGain = inactiveGain; inactiveGain = tG
}

async function crossfadeTo(url, fadeSec = 3.0, startTime = 0){
  if (!url) return
  await _resumeAudioCtx()
  try {
    inactiveEl.src = url
    inactiveGain.gain.cancelScheduledValues(audioCtx.currentTime)
    inactiveGain.gain.setValueAtTime(0.0, audioCtx.currentTime)

    // ensure we seek to requested start time after metadata is ready
    const startAndPlay = async () => {
      try {
        if (startTime && typeof inactiveEl.currentTime !== 'undefined') {
          try { inactiveEl.currentTime = startTime } catch (__) {}
        }
        await inactiveEl.play()
      } catch (e) {
        try { inactiveEl.play() } catch (__) {}
      }
    }

    if (inactiveEl.readyState >= 1) {
      await startAndPlay()
    } else {
      await new Promise((resolve) => {
        inactiveEl.addEventListener('loadedmetadata', async () => { await startAndPlay(); resolve() }, { once: true })
      })
    }

    const now = audioCtx.currentTime
    inactiveGain.gain.linearRampToValueAtTime(1.0, now + fadeSec)
    activeGain.gain.linearRampToValueAtTime(0.0, now + fadeSec)
    // after fade completes, pause previous and swap
    setTimeout(() => {
      try { activeEl.pause(); activeEl.currentTime = 0 } catch (e) {}
      _swapActive()
      isPlaying = true
      partyStatus.textContent = 'Playing'
    }, Math.round((fadeSec + 0.05) * 1000))
  } catch (e) { console.warn('crossfadeTo failed', e) }
}

// singalong UI removed

  async function playPath(path){
    if (!path) return
    const url = filePathToFileUrl(path)
    if (!url) return
    const startTime = 0
    if (isPlaying) {
      // crossfade into the new track
      await crossfadeTo(url, 2.0, startTime)
    } else {
      // immediate play
      await playImmediate(url, startTime)
    }
  }

// receive updates from main
safeOnAPI('onPartyState', (s) => {
  lastParty = s
  try {
    energyValue.textContent = Math.round(s.energy || 0)
    energyTrend.textContent = s.trend || 'stable'
    partyStatus.textContent = s.status || 'idle'
    currentVibe.textContent = s.vibe || currentVibe.textContent || '—'
    nowTrack.textContent = s.now_track || '(none)'
    nextTrack.textContent = s.next_track || '(queued)'
    // keep slider showing current target when no user drag
    const curSlider = Number(energySlider.value)
    if (Math.abs((s.target || 0) - curSlider) > 3) {
      energySlider.value = Math.round(s.target || 0)
      sliderValue.textContent = `(${Math.round(s.target||0)})`
    }
  } catch (e) { /* ignore */ }
})

// when server returns a selection result, update UI and notify memory of play-start
safeOnAPI('onPartyTrack', async (obj) => {
  try {
    if (!obj) return
    // obj expected shape: { best: <path>, scored: [{path, score, components}, ...] }
    const best = obj.best
    if (best) {
      nowTrack.textContent = best
      // next track: pick second-best if available
      const next = (Array.isArray(obj.scored) && obj.scored.length > 1) ? obj.scored[1].path : '(queued)'
      nextTrack.textContent = next

      // if currently playing, switch audio to new track
      try {
        if (best) {
          await playPath(best)
        }
      } catch (e) { console.warn('onPartyTrack playback failed', e) }

      // notify playback/memory: record play-start (played_at in seconds)
      try { await window.api.recordPlay({ path: best, played_at: Date.now() / 1000.0, duration: null }) } catch (e) { console.warn('recordPlay failed', e) }
    }
  } catch (e) { /* ignore */ }
})

// scan progress events
safeOnAPI('onProgress', (data) => {
  try {
    if (!data) return
    scanTotal = data.count || scanTotal
    const statusEl = document.getElementById('scanStatus')
    if (statusEl) statusEl.textContent = `Found ${scanTotal} tracks — preparing scan...`
  } catch (e) {}
})

safeOnAPI('onDone', (data) => {
  try {
    scanTotal = data.count || scanTotal
    const statusEl = document.getElementById('scanStatus')
    if (statusEl) statusEl.textContent = `Scanning ${scanTotal} tracks…`
  } catch (e) {}
})

safeOnAPI('onMusicscanProgress', (obj) => {
  try {
    // Count unique per-path messages. musicscan may emit multiple JSON
    // messages per file (full features, compact per-path summary, etc.) and
    // may also emit error objects for bad files. Track seen paths so we
    // increment the progress only once per file and avoid stalling when a
    // malformed file emits an error message without feature keys.
    const progressEl = document.getElementById('scanProgress')
    const statusEl = document.getElementById('scanStatus')
    let counted = false
    if (obj && obj.path) {
      const p = obj.path
      if (!scannedPaths.has(p)) {
        scannedPaths.add(p)
        scannedCount += 1
        counted = true
      }
      const pct = scanTotal ? Math.round((scannedCount / scanTotal) * 100) : 0
      if (progressEl) progressEl.value = pct
      if (statusEl) statusEl.textContent = `${scannedCount}/${scanTotal} ${counted ? '– ' + (obj._cached ? '(cached) ' : '') + obj.path : ''}`
    } else {
      // non-path JSON (clusters/compat) — ignore for counting
      const pct = scanTotal ? Math.round((scannedCount / scanTotal) * 100) : 0
      if (progressEl) progressEl.value = pct
    }
  } catch (e) {}
})

safeOnAPI('onMusicscanComplete', (res) => {
  try {
    scanning = false; scanComplete = true
    importFilesBtn.textContent = 'Imported'
    const statusEl = document.getElementById('scanStatus')
    if (statusEl) statusEl.textContent = `Scan complete — ${scannedCount} tracks processed`
    startPartyBtn.disabled = false
  } catch (e) {}
})

// Guest events: display QR and list of requests
safeOnAPI('onGuestUrl', (obj) => {
  try {
    const url = (obj && obj.url) ? obj.url : (obj && obj.url === undefined ? obj : null)
    if (!url) return
    guestUrlEl.textContent = url
    // use Google Charts QR fallback
    try {
      guestQREl.src = 'https://chart.googleapis.com/chart?cht=qr&chs=220x220&chl=' + encodeURIComponent(url)
    } catch (e) {
      // ignore
    }
  } catch (e) {}
})

function renderGuestRequests(list) {
  try {
    if (!Array.isArray(list) || list.length === 0) {
      guestListEl.textContent = 'No requests yet.'
      return
    }
    guestListEl.innerHTML = ''
    list.forEach(r => {
      const d = document.createElement('div')
      d.style.display = 'flex'
      d.style.justifyContent = 'space-between'
      d.style.alignItems = 'center'
      d.style.marginBottom = '6px'
      const left = document.createElement('div')
      left.innerHTML = `<div style="font-weight:600">${(r.title||r.path)}</div><div style="font-size:12px;color:var(--muted)">upvotes: ${r.upvotes} • energy:${r.energy}</div>`
      const right = document.createElement('div')
      const injectBtn = document.createElement('button')
      injectBtn.textContent = 'Inject'
      injectBtn.style.marginLeft = '8px'
      injectBtn.onclick = async () => {
        try {
          await window.api.injectRequest(r.id)
        } catch (e) { console.warn('inject failed', e) }
      }
      right.appendChild(injectBtn)
      d.appendChild(left)
      d.appendChild(right)
      guestListEl.appendChild(d)
    })
  } catch (e) { guestListEl.textContent = 'err' }
}

safeOnAPI('onGuestRequests', (arr) => {
  renderGuestRequests(arr)
})

safeOnAPI('onGuestInject', (obj) => {
  try {
    // briefly highlight and set nextTrack
    if (obj && obj.path) {
      nextTrack.textContent = obj.path
    }
  } catch (e) {}
})

// Human-friendly feedback messages (from Python party server)
safeOnAPI('onPartyFeedback', (msg) => {
  try {
    if (!msg) return
    if (feedbackTimer) clearTimeout(feedbackTimer)
    feedbackMessageEl.textContent = msg
    feedbackTimer = setTimeout(() => { feedbackMessageEl.textContent = 'Ready to steer the party' }, 6000)
  } catch (e) {}
})

// Host-choice prompt disabled: server no longer emits 'ask_choice'.

// Advanced panel toggle
if (advancedToggle) {
  advancedToggle.addEventListener('click', () => {
    if (advancedPanel.classList.contains('hidden')) {
      advancedPanel.classList.remove('hidden')
      advancedToggle.textContent = 'Advanced ▴'
    } else {
      advancedPanel.classList.add('hidden')
      advancedToggle.textContent = 'Advanced ▾'
    }
  })
}

if (tempoBiasEl && tempoBiasValueEl) {
  tempoBiasEl.addEventListener('input', (e) => { tempoBiasValueEl.textContent = `(${e.target.value})` })
}

if (saveAdvancedBtn) {
  saveAdvancedBtn.addEventListener('click', async () => {
    const cfg = { genre_filter: (genreFilterEl && genreFilterEl.value) || '', hide_explicit: !!(explicitToggleEl && explicitToggleEl.checked), tempo_bias: Number(tempoBiasEl && tempoBiasEl.value) || 0 }
    try {
      await window.api.setAdvanced(cfg)
      if (feedbackTimer) clearTimeout(feedbackTimer)
      feedbackMessageEl.textContent = 'Advanced settings saved'
      feedbackTimer = setTimeout(() => { feedbackMessageEl.textContent = 'Ready to steer the party' }, 3000)
    } catch (e) {
      console.warn('setAdvanced failed', e)
    }
  })
}

// Failure prompt response buttons
// Playback control handlers
if (playBtn) playBtn.addEventListener('click', async () => {
  try {
    // if no audio loaded, try to play nowTrack or nextTrack
    let path = (nowTrack && nowTrack.textContent && nowTrack.textContent !== '(none)') ? nowTrack.textContent : null
    if (!path) path = (nextTrack && nextTrack.textContent && nextTrack.textContent !== '(queued)') ? nextTrack.textContent : null
    if (!path) { alert('No track selected. Import files and scan first.'); return }
    try {
      await playPath(path)
      try { await window.api.recordPlay({ path: path, played_at: Date.now() / 1000.0 }) } catch (e) {}
      await window.api.playbackPlay({ path: path, played_at: Date.now() / 1000.0 })
    } catch (e) { console.warn('play failed', e) }
  } catch (e) { console.warn('play failed', e) }
})
if (pauseBtn) pauseBtn.addEventListener('click', async () => {
  try { if (activeEl) activeEl.pause(); isPlaying = false; await window.api.playbackPause({}); partyStatus.textContent = 'Paused' } catch (e) { console.warn('pause failed', e) }
})
if (skipBtn) skipBtn.addEventListener('click', async () => {
  try {
    // stop current audio
    try { if (activeEl) activeEl.pause() } catch (__) {}
    // singalong monitoring removed
    const res = await window.api.playbackSkip({ played_at: Date.now() / 1000.0 })
    let playPath = null
    try {
      if (res && res.now) playPath = res.now
      else if (res && res.next) playPath = res.next
    } catch (e) { playPath = null }
    if (playPath) {
      try {
        await playPath(playPath)
        isPlaying = true
        nowTrack.textContent = playPath
        nextTrack.textContent = '(queued)'
        try { await window.api.recordPlay({ path: playPath, played_at: Date.now() / 1000.0 }) } catch (e) {}
      } catch (e) { console.warn('skip playback failed', e) }
    } else {
      nowTrack.textContent = '(none)'
      nextTrack.textContent = '(queued)'
      partyStatus.textContent = 'Idle'
      isPlaying = false
    }
  } catch (e) { console.warn('skip failed', e) }
})
function filePathToFileUrl(p) {
  if (!p) return null
  try {
    let r = p.replace(/\\\\/g, '/').replace(/\\/g, '/')
    if (!r.startsWith('/')) {
      // windows drive path
      return 'file:///' + encodeURI(r)
    }
    return 'file://' + encodeURI(r)
  } catch (e) {
    return null
  }
}

function _onAudioEnded(){
  try {
    isPlaying = false
    // singalong monitoring removed
    // attempt to play queued next
    const nxt = (nextTrack && nextTrack.textContent && nextTrack.textContent !== '(queued)') ? nextTrack.textContent : null
    if (nxt) {
      const url = filePathToFileUrl(nxt)
      if (url) {
        (async () => {
          try {
            await playPath(nxt)
            isPlaying = true
            nowTrack.textContent = nxt
            nextTrack.textContent = '(queued)'
            try { await window.api.recordPlay({ path: nxt, played_at: Date.now() / 1000.0 }) } catch (e) {}
          } catch (e) { console.warn('play next on ended failed', e) }
        })()
      }
    } else {
      partyStatus.textContent = 'Idle'
    }
  } catch (e) {}
}
try { audioA.addEventListener('ended', _onAudioEnded); audioB.addEventListener('ended', _onAudioEnded) } catch (e) {}

// initialize
document.addEventListener('DOMContentLoaded', () => {
  // ensure UI reflects initial party state (safe subscribe)
  safeOnAPI('onPartyState', (s) => {})
  // request no-op: main will broadcast periodically
  // check for an existing cached musicscan DB and inform user
  (async () => {
    try {
      const hasCache = await window.api.hasScanCache()
      const statusEl = document.getElementById('scanStatus')
      if (hasCache && statusEl) {
        statusEl.textContent = 'Found existing scan cache (musicscan.db) — import to reuse or rescan.'
        importFilesBtn.textContent = 'Import (cached)'
      }
    } catch (e) {}
  })()
  // populate last-scanned folder info after preload is ready
  ;(async () => {
    try {
      if (window.api && typeof window.api.getLastScan === 'function') {
        const st = await window.api.getLastScan()
        if (st && st.last_folder) {
          selectedFolder = st.last_folder
          if (importFilesBtn) importFilesBtn.textContent = 'Using last scanned folder'
          const statusEl = document.getElementById('scanStatus')
          if (statusEl) statusEl.textContent = `Last scanned: ${st.last_folder} — ${st.count || 0} tracks`
          scanTotal = 0; scannedCount = 0; scanning = false; scanComplete = false
          scannedPaths = new Set()
          // restore NL prompt if present in scan state and wire inputs
          try {
            const nlEl = document.getElementById('nlPrompt')
            const obEl = document.getElementById('onboardPrompt')
            const applyPrompt = async (v) => {
              try { if (window.api && typeof window.api.setPrompt === 'function') await window.api.setPrompt(v) } catch (e) {}
              try { if (window.api && typeof window.api.setAdvanced === 'function') await window.api.setAdvanced({ nl_prompt: v }) } catch (e) {}
            }
            if (st && st.last_prompt) {
              if (nlEl) nlEl.value = st.last_prompt
              if (obEl) obEl.value = st.last_prompt
              // notify server of the prompt so selection respects it
              try { if (window.api && typeof window.api.setAdvanced === 'function') window.api.setAdvanced({ nl_prompt: st.last_prompt }) } catch (e) {}
            }
            const wire = (el) => {
              if (!el) return
              let promptTimer = null
              el.addEventListener('input', (ev) => {
                const v = (ev && ev.target && ev.target.value) ? ev.target.value : ''
                if (promptTimer) clearTimeout(promptTimer)
                promptTimer = setTimeout(async () => { await applyPrompt(v) }, 600)
              })
              el.addEventListener('keydown', (ev) => {
                if (ev && ev.key === 'Enter' && !ev.shiftKey && !ev.metaKey && !ev.ctrlKey) {
                  ev.preventDefault()
                  const v = el.value || ''
                  if (promptTimer) clearTimeout(promptTimer)
                  ;(async () => { await applyPrompt(v) })()
                }
              })
            }
            wire(nlEl)
            wire(obEl)
          } catch (e) {}
        }
      }
    } catch (e) {}
  })()
})
