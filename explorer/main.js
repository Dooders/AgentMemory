const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');

// Enable hot reloading in development
if (process.argv.includes('--dev')) {
  try {
    require('electron-reloader')(module, {
      debug: true,
      watchRenderer: true
    });
  } catch (_) { console.log('Error hot reloading'); }
}

// Get the default memory directory
function getDefaultMemoryDir() {
  // Point to the specific agent_farm_memories.json file
  return path.join(__dirname, '..', 'validation', 'memory_samples', 'agent_farm_memories.json');
}

// Load memory files from a directory
async function loadMemoryFiles(filePath) {
  try {
    console.log('Attempting to load file:', filePath);
    if (!fs.existsSync(filePath)) {
      console.error('File not found:', filePath);
      return { error: 'Memory file not found' };
    }

    try {
      console.log('Reading file...');
      const raw = fs.readFileSync(filePath, 'utf8');
      console.log('File read, size:', raw.length);
      console.log('Parsing JSON...');
      const data = JSON.parse(raw);
      console.log('JSON parsed successfully, data type:', typeof data);
      return { canceled: false, contents: [{ path: filePath, data }] };
    } catch (err) {
      console.error('Error processing file:', err);
      return { error: `Failed to parse ${filePath}: ${err.message}` };
    }
  } catch (err) {
    console.error('Error in loadMemoryFiles:', err);
    return { error: err.message };
  }
}

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  // Open DevTools in development
  win.webContents.openDevTools();

  // Handle renderer process errors
  win.webContents.on('did-fail-load', (event, errorCode, errorDescription) => {
    console.error('Failed to load:', errorCode, errorDescription);
  });

  win.loadFile(path.join(__dirname, 'renderer', 'index.html'));
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

// IPC handlers

ipcMain.handle('dialog:openFiles', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog({
    properties: ['openFile', 'multiSelections'],
    filters: [{ name: 'JSON', extensions: ['json', 'jsonl'] }]
  });
  if (canceled) return { canceled: true };
  const contents = [];
  for (const filePath of filePaths) {
    try {
      const raw = fs.readFileSync(filePath, 'utf8');
      const data = JSON.parse(raw);
      contents.push({ path: filePath, data });
    } catch (err) {
      contents.push({ path: filePath, error: err.message });
    }
  }
  return { canceled: false, contents };
});

ipcMain.handle('loadDefaultMemories', async () => {
  console.log('loadDefaultMemories called');
  const memoryDir = getDefaultMemoryDir();
  console.log('Default memory path:', memoryDir);
  const result = await loadMemoryFiles(memoryDir);
  console.log('Load result:', result);
  return result;
}); 