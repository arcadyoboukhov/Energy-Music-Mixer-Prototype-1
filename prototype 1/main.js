const { app, BrowserWindow, ipcMain, dialog } = require('electron')
const path = require('path')
const fs = require('fs').promises
const fsSync = require('fs')
const { spawn } = require('child_process')

const SCAN_STATE_FILE = path.join(__dirname, 'scan_state.json')

async function readScanState() {
  try {
    const txt = await fs.readFile(SCAN_STATE_FILE, 'utf8')
    return JSON.parse(txt)
  } catch (e) {
    return null
  }
}

async function writeScanState(folder, count, bytes, prompt) {
  // merge with existing state so we don't clobber other keys like last_prompt
  try {
    let existing = {}
    try { const txt = await fs.readFile(SCAN_STATE_FILE, 'utf8'); existing = JSON.parse(txt) } catch (e) { existing = {} }
    const obj = Object.assign({}, existing)
    obj.last_folder = folder
    obj.count = Number(count) || 0
    obj.bytes = Number(bytes) || 0
    obj.last_scanned_at = (new Date()).toISOString()
    if (typeof prompt !== 'undefined') obj.last_prompt = String(prompt)
    try { await fs.writeFile(SCAN_STATE_FILE, JSON.stringify(obj), 'utf8') } catch (e) { console.error('failed to write scan state', e) }
  } catch (e) {
    console.error('writeScanState failed', e)
  }
}

// helper: determine python command (prefer venv)
function getPythonCmd() {
  let pythonCmd = 'python'
  try {
    const venvWin = path.join(__dirname, '.venv', 'Scripts', 'python.exe')
    const venvPosix = path.join(__dirname, '.venv', 'bin', 'python')
    if (fsSync.existsSync(venvWin)) pythonCmd = venvWin
    else if (fsSync.existsSync(venvPosix)) pythonCmd = venvPosix
  } catch (e) {}
  return pythonCmd
}

// currentScan controller used to pause/resume a running scan
let currentScan = null

ipcMain.handle('check-folder-cache', async (event, folderPath) => {
  if (!folderPath) return { error: 'missing_folder' }
  const pythonCmd = getPythonCmd()
  const checker = path.join(__dirname, 'tools', 'check_musicscan_cache.py')
  try {
    const checkProc = spawn(pythonCmd, [checker, folderPath, path.join(__dirname, 'musicscan.db')])
    let out = ''
    let err = ''
    checkProc.stdout.on('data', (d) => { out += d.toString() })
    checkProc.stderr.on('data', (d) => { err += d.toString(); try { event.sender.send('musicscan-stderr', d.toString()) } catch (e) {} })
    const code = await new Promise((resolve) => { checkProc.on('close', resolve) })
    try { return JSON.parse((out || '').trim() || '{}') } catch (e) { return { error: 'parse_failed', stdout: out, stderr: err, code } }
  } catch (e) {
    return { error: 'checker_failed', message: String(e) }
  }
})

ipcMain.handle('pause-scan', async () => {
  if (!currentScan) return { ok: false, reason: 'no_scan' }
  currentScan.paused = true
  return { ok: true }
})

ipcMain.handle('resume-scan', async () => {
  if (!currentScan) return { ok: false, reason: 'no_scan' }
  if (!currentScan.paused) return { ok: false, reason: 'not_paused' }
  currentScan.paused = false
  try {
    // resolve any waiters
    (currentScan.resumeResolvers || []).forEach(r => { try { r() } catch (__) {} })
    currentScan.resumeResolvers = []
  } catch (e) {}
  return { ok: true }
})

// Party server process bridge (spawned Python helper)
let partyProc = null
// pending resolvers for select-initial requests
let pendingSelectResolvers = []
// pending resolvers for playback-skip requests
let pendingSkipResolvers = []
function sendToPartyServer(obj) {
  try {
    if (partyProc && partyProc.stdin && !partyProc.killed) {
      partyProc.stdin.write(JSON.stringify(obj) + '\n')
      return true
    }
  } catch (e) {
    console.error('sendToPartyServer failed', e)
  }
  return false
}

function createWindow() {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js')
    }
  })
  win.loadFile('index.html')
}

app.whenReady().then(() => {
  createWindow()
  // spawn the Python party server that will manage PartyState and memory
  try {
  // prefer project venv python when available
  let pythonCmd = 'python'
  const venvWin = path.join(__dirname, '.venv', 'Scripts', 'python.exe')
  const venvPosix = path.join(__dirname, '.venv', 'bin', 'python')
  if (fsSync.existsSync(venvWin)) pythonCmd = venvWin
  else if (fsSync.existsSync(venvPosix)) pythonCmd = venvPosix
  partyProc = spawn(pythonCmd, [path.join(__dirname, 'party_server.py')])
    partyProc.stderr.on('data', (d) => {
      const s = d.toString()
      console.error('party stderr:', s)
      for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-stdout', s) } catch (e) {} }
    })

    let pbuf = ''
    partyProc.stdout.on('data', (d) => {
      pbuf += d.toString()
      let idx
      while ((idx = pbuf.indexOf('\n')) !== -1) {
        const line = pbuf.slice(0, idx).trim()
        pbuf = pbuf.slice(idx + 1)
        if (!line) continue
        try {
          const obj = JSON.parse(line)
          if (obj.type === 'party_state') {
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-state', obj.state) } catch (e) {} }
          } else if (obj.type === 'select_result') {
            try { pendingSelectResolvers.forEach(r => { try { r(obj) } catch (e) {} }) } catch (e) {}
            pendingSelectResolvers = []
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-track', obj) } catch (e) {} }
          } else if (obj.type === 'skip_result') {
            try { pendingSkipResolvers.forEach(r => { try { r(obj) } catch (e) {} }) } catch (e) {}
            pendingSkipResolvers = []
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-skip', obj) } catch (e) {} }
          } else if (obj.type === 'feedback') {
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-feedback', obj.message) } catch (e) {} }
          } else if (obj.type === 'ask_choice') {
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-ask', obj) } catch (e) {} }
          } else if (obj.type === 'guest_ready') {
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-guest', obj) } catch (e) {} }
          } else if (obj.type === 'guest_requests') {
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-guest-requests', obj.requests) } catch (e) {} }
          } else if (obj.type === 'guest_inject') {
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-guest-inject', obj) } catch (e) {} }
          } else {
            for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-log', obj) } catch (e) {} }
          }
        } catch (e) {
          for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-stdout', line) } catch (e) {} }
        }
      }
    })

    partyProc.on('close', (code) => {
      console.log('party server exited', code)
      for (const w of BrowserWindow.getAllWindows()) { try { w.webContents.send('party-exit', { code }) } catch (e) {} }
      partyProc = null
    })
  } catch (e) {
    console.error('failed to spawn party_server', e)
  }
})

// check for existing musicscan DB cache (quick existence check)
ipcMain.handle('has-scan-cache', async () => {
  try {
    const dbPath = path.join(__dirname, 'musicscan.db')
    return fsSync.existsSync(dbPath)
  } catch (e) {
    return false
  }
})

ipcMain.handle('select-folder', async () => {
  const res = await dialog.showOpenDialog({ properties: ['openDirectory'] })
  if (res.canceled || res.filePaths.length === 0) return null
  return res.filePaths[0]
})

ipcMain.handle('get-last-scan', async () => {
  try {
    const st = await readScanState()
    return st || null
  } catch (e) {
    return null
  }
})

ipcMain.handle('set-prompt', async (event, prompt) => {
  try {
    const prev = await readScanState().catch(() => null) || {}
    const folder = prev.last_folder || ''
    const count = Number(prev.count || 0)
    const bytes = Number(prev.bytes || 0)
    // write prompt into scan state (preserving other fields)
    try { await writeScanState(folder, count, bytes, prompt) } catch (e) { /* ignore */ }
    // also forward to party server as an advanced filter so selector can use it
    try { sendToPartyServer({ cmd: 'set_advanced', params: { nl_prompt: prompt } }) } catch (e) {}
    return { ok: true }
  } catch (e) {
    return { error: String(e) }
  }
})

ipcMain.handle('get-features', async (event, filePath) => {
  if (!filePath) return { error: 'missing_path' }
  const pythonCmd = getPythonCmd()
  const helper = path.join(__dirname, 'tools', 'get_musicscan_feature.py')
  try {
    const db = path.join(__dirname, 'musicscan.db')
    const p = spawn(pythonCmd, [helper, db, filePath])
    let out = ''
    let err = ''
    p.stdout.on('data', (d) => { out += d.toString() })
    p.stderr.on('data', (d) => { err += d.toString() })
    const code = await new Promise((resolve) => { p.on('close', resolve) })
    try { return JSON.parse((out || '').trim() || '{}') } catch (e) { return { error: 'parse_failed', stdout: out, stderr: err, code } }
  } catch (e) {
    return { error: 'helper_failed', message: String(e) }
  }
})

ipcMain.handle('start-scan', async (event, folderPath) => {
  // if no folder specified, fall back to last scanned folder (remembered)
  if (!folderPath) {
    const st = await readScanState().catch(() => null)
    if (st && st.last_folder) folderPath = st.last_folder
    else return false
  }
  // prevent concurrent scans
  if (currentScan) {
    try { event.sender.send('musicscan-stderr', 'scan already in progress') } catch (e) {}
    return false
  }
  // initialize scan controller for pause/resume
  currentScan = { paused: false, resumeResolvers: [], folder: folderPath }
  // track files we've attempted to repair during this scan to avoid retries
  const repairAttempted = new Set()
  const exts = new Set(['.mp3', '.flac'])
  // Phase 1: fast counting pass (no Python) so UI gets instant totals
  let count = 0
  let totalBytes = 0
  let lastSent = Date.now()

  async function walkCount(dir) {
    let entries
    try { entries = await fs.readdir(dir, { withFileTypes: true }) }
    catch (e) { return }
    for (const e of entries) {
      // respect pause requests during the counting pass
      if (currentScan && currentScan.paused) {
        // await resume
        await new Promise((resolve) => {
          currentScan.resumeResolvers = currentScan.resumeResolvers || []
          currentScan.resumeResolvers.push(resolve)
        })
      }
      const full = path.join(dir, e.name)
      if (e.isDirectory()) {
        await walkCount(full)
      } else if (e.isFile()) {
        const ext = path.extname(e.name).toLowerCase()
        if (exts.has(ext)) {
          count += 1
          try { const st = await fs.stat(full); totalBytes += st.size } catch (_) {}
          const now = Date.now()
          if (now - lastSent > 200) {
            lastSent = now
            event.sender.send('scan-progress', { count, bytes: totalBytes })
          }
        }
      }
    }
  }

  (async () => {
    // quick pass to compute totals
    await walkCount(folderPath)
    event.sender.send('scan-done', { count, bytes: totalBytes })

    // fast-skip: if we previously scanned this exact folder and the
    // stored count/bytes match current counts, assume nothing changed
    // and skip the expensive cache check and scan.
    try {
      const prev = await readScanState().catch(() => null)
      const dbPath = path.join(__dirname, 'musicscan.db')
      if (prev && prev.last_folder === folderPath && Number(prev.count) === Number(count) && Number(prev.bytes) === Number(totalBytes) && fsSync.existsSync(dbPath)) {
        try { await writeScanState(folderPath, count, totalBytes) } catch (e) {}
        try { event.sender.send('musicscan-complete', { code: 0, cached: true, count: count }) } catch (e) {}
        currentScan = null
        return
      }
    } catch (e) {
      // ignore and continue to full check
    }

    // Phase 2: if all files are already cached in musicscan.db, skip the
    // expensive per-file feature extraction. Otherwise stream paths to
    // python for detailed processing (separate walk).
    // prefer project venv python when available
    let pythonCmd = 'python'
    const venvWinLocal = path.join(__dirname, '.venv', 'Scripts', 'python.exe')
    const venvPosixLocal = path.join(__dirname, '.venv', 'bin', 'python')
    if (fsSync.existsSync(venvWinLocal)) pythonCmd = venvWinLocal
    else if (fsSync.existsSync(venvPosixLocal)) pythonCmd = venvPosixLocal
    const repairHelper = path.join(__dirname, 'tools', 'repair_musicscan_entry.py')

    // check cache helper
    const checker = path.join(__dirname, 'tools', 'check_musicscan_cache.py')
    try {
      const checkProc = spawn(pythonCmd, [checker, folderPath, path.join(__dirname, 'musicscan.db')])
      let checkOut = ''
      let checkErr = ''
      checkProc.stdout.on('data', (d) => { checkOut += d.toString() })
      checkProc.stderr.on('data', (d) => { checkErr += d.toString(); event.sender.send('musicscan-stderr', d.toString()) })
      const code = await new Promise((resolve) => { checkProc.on('close', resolve) })
      try {
        const res = JSON.parse((checkOut || '').trim() || '{}')
        if (res && res.all_cached) {
          // fast path: persist scan state and emit a single completion event
          try { await writeScanState(folderPath, res.count || count, totalBytes) } catch (e) {}
          try { event.sender.send('musicscan-complete', { code: 0, cached: true, count: res.count || count }) } catch (e) {}
          // clear controller and return
          currentScan = null
          return
        }
      } catch (e) {
        // fallthrough to full scan
      }
    } catch (e) {
      // if the checker failed, fall back to full scan
    }

    const py = spawn(pythonCmd, [path.join(__dirname, 'musicscan.py')])
    py.stderr.on('data', (d) => { event.sender.send('musicscan-stderr', d.toString()) })

    let stdoutBuf = ''
    py.stdout.on('data', (d) => {
      stdoutBuf += d.toString()
      let idx
      while ((idx = stdoutBuf.indexOf('\n')) !== -1) {
        const line = stdoutBuf.slice(0, idx).trim()
        stdoutBuf = stdoutBuf.slice(idx + 1)
        if (!line) continue
        try {
          const obj = JSON.parse(line)
          event.sender.send('musicscan-progress', obj)
          try { console.log(JSON.stringify(obj)) } catch (__) {}

          // If a single-file error is reported, attempt a per-file repair
          if (obj && obj.error && obj.path && !repairAttempted.has(obj.path)) {
            repairAttempted.add(obj.path)
            try {
              const rproc = spawn(pythonCmd, [repairHelper, path.join(__dirname, 'musicscan.db'), obj.path])
              let rout = ''
              let rerr = ''
              rproc.stdout.on('data', (rd) => { rout += rd.toString(); try { event.sender.send('musicscan-stderr', rd.toString()) } catch (__) {} })
              rproc.stderr.on('data', (rd) => { rerr += rd.toString(); event.sender.send('musicscan-stderr', rd.toString()) })
              rproc.on('close', (rcode) => {
                try {
                  const parsed = JSON.parse((rout || '').trim() || '{}')
                  if (parsed && !parsed.error) {
                    event.sender.send('musicscan-progress', parsed)
                  } else {
                    event.sender.send('musicscan-stderr', 'repair failed for ' + obj.path + ' - ' + (parsed && parsed.error ? parsed.error : (rerr || rcode)))
                  }
                } catch (e) {
                  event.sender.send('musicscan-stderr', 'repair parse failed for ' + obj.path + ': ' + e + ' stdout:' + rout + ' stderr:' + rerr)
                }
              })
            } catch (e) {
              try { event.sender.send('musicscan-stderr', 'repair spawn failed: ' + String(e)) } catch (__) {}
            }
          }
        } catch (e) {
          event.sender.send('musicscan-stdout-line', line)
        }
      }
    })

    py.on('close', (code) => {
      try { writeScanState(folderPath, count, totalBytes).catch(() => {}) } catch (e) {}
      try { event.sender.send('musicscan-complete', { code }) } catch (e) {}
      // clear controller
      currentScan = null
    })

    function writeOrWait(s) {
      return new Promise((resolve) => {
        try {
          const ok = py.stdin.write(s)
          if (ok) return resolve()
          py.stdin.once('drain', resolve)
        } catch (e) {
          // swallow write errors (EPIPE) and report to renderer
          event.sender.send('musicscan-stderr', 'stdin write error: ' + String(e))
          return resolve()
        }
      })
    }

    // listen for python stdin errors (EPIPE etc.) to avoid uncaught exceptions
    try {
      py.stdin.on('error', (e) => {
        event.sender.send('musicscan-stderr', 'py.stdin error: ' + String(e))
      })
    } catch (e) {
      // ignore if stdin not available
    }

    async function walkAndStream(dir) {
      let entries
      try { entries = await fs.readdir(dir, { withFileTypes: true }) }
      catch (e) { return }
      for (const e of entries) {
        const full = path.join(dir, e.name)
        // allow pausing between files
        if (currentScan && currentScan.paused) {
          await new Promise((resolve) => {
            currentScan.resumeResolvers = currentScan.resumeResolvers || []
            currentScan.resumeResolvers.push(resolve)
          })
        }
        if (!currentScan) return
        if (e.isDirectory()) {
          await walkAndStream(full)
        } else if (e.isFile()) {
          const ext = path.extname(e.name).toLowerCase()
          if (exts.has(ext)) {
            // stream path to python process (newline separated)
            try {
              await writeOrWait(full + '\n')
            } catch (e) {
              // ignore write errors; python may have died
            }
          }
        }
      }
    }

    try {
      await walkAndStream(folderPath)
      try { py.stdin.end() } catch (e) {}
    } finally {
      // ensure controller cleared if stream loop exits normally
      // (py.on('close') will also clear when process exits)
      // resolve any remaining resume waiters to avoid leaks
      try {
        if (currentScan && currentScan.resumeResolvers) {
          currentScan.resumeResolvers.forEach(r => { try { r() } catch (__) {} })
          currentScan.resumeResolvers = []
        }
      } catch (e) {}
    }
  })()

  return true
})

// Party IPC handlers (forward to Python party server when available)

ipcMain.handle('start-party', (event, opts) => {
  const ok = sendToPartyServer({ cmd: 'start_party', vibe: (opts && opts.vibe) || null })
  return { ok: !!ok }
})

ipcMain.handle('set-energy-target', (event, val) => {
  const v = Number(val) || 0
  const ok = sendToPartyServer({ cmd: 'set_energy_target', value: Math.max(0, Math.min(100, Math.round(v))) })
  return { ok: !!ok }
})

ipcMain.handle('adjust-familiarity-bias', (event, delta) => {
  const d = Number(delta) || 0
  const ok = sendToPartyServer({ cmd: 'adjust_familiarity_bias', delta: d })
  return { ok: !!ok }
})

// record-play: playback path should call this when a track starts playing
ipcMain.handle('record-play', (event, params) => {
  const ok = sendToPartyServer({ cmd: 'record_play', params: params || {} })
  return { ok: !!ok }
})

// Playback controls forwarded to party server
ipcMain.handle('playback-play', (event, params) => {
  const ok = sendToPartyServer({ cmd: 'playback_play', params: params || {} })
  return { ok: !!ok }
})

ipcMain.handle('playback-pause', (event, params) => {
  const ok = sendToPartyServer({ cmd: 'playback_pause', params: params || {} })
  return { ok: !!ok }
})

ipcMain.handle('playback-skip', (event, params) => {
  return new Promise((resolve) => {
    let timedOut = false
    const resolver = (obj) => {
      if (timedOut) return
      timedOut = true
      resolve(obj)
    }
    pendingSkipResolvers.push(resolver)
    const ok = sendToPartyServer({ cmd: 'playback_skip', params: params || {} })
    if (!ok) {
      // remove resolver and resolve null immediately
      pendingSkipResolvers = pendingSkipResolvers.filter(r => r !== resolver)
      resolve(null)
      return
    }
    // safety timeout
    setTimeout(() => {
      if (!timedOut) {
        timedOut = true
        pendingSkipResolvers = pendingSkipResolvers.filter(r => r !== resolver)
        resolve(null)
      }
    }, 5000)
  })
})

ipcMain.handle('inject-request', (event, requestId) => {
  const ok = sendToPartyServer({ cmd: 'inject_request', id: requestId })
  return { ok: !!ok }
})

// advanced settings from UI
ipcMain.handle('set-advanced', (event, opts) => {
  const ok = sendToPartyServer({ cmd: 'set_advanced', params: opts || {} })
  return { ok: !!ok }
})

// failure handling response from user
ipcMain.handle('failure-response', (event, choice) => {
  const ok = sendToPartyServer({ cmd: 'failure_response', choice: choice })
  return { ok: !!ok }
})

ipcMain.handle('select-initial', (event) => {
  return new Promise((resolve) => {
    // resolver will be called when a 'select_result' is emitted from the party server
    let timedOut = false
    const resolver = (obj) => {
      if (timedOut) return
      timedOut = true
      resolve(obj)
    }
    pendingSelectResolvers.push(resolver)
    // send select_initial command to the server
    const ok = sendToPartyServer({ cmd: 'select_initial' })
    if (!ok) {
      // remove resolver and resolve null immediately
      pendingSelectResolvers = pendingSelectResolvers.filter(r => r !== resolver)
      resolve(null)
      return
    }
    // safety timeout
    setTimeout(() => {
      if (!timedOut) {
        timedOut = true
        pendingSelectResolvers = pendingSelectResolvers.filter(r => r !== resolver)
        resolve(null)
      }
    }, 5000)
  })
})

app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
