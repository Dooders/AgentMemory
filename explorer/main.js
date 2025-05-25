const { app, BrowserWindow, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs').promises;

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
    try {
      await fs.access(filePath);
    } catch {
      console.error('File not found:', filePath);
      return { error: 'Memory file not found' };
    }

    try {
      console.log('Reading file...');
      const raw = await fs.readFile(filePath, 'utf8');
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

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // In development, load from Vite dev server
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173');
    mainWindow.webContents.openDevTools();
  } else {
    // In production, load the built files
    mainWindow.loadFile(path.join(__dirname, 'dist', 'index.html'));
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Handle file opening
ipcMain.handle('openFile', async () => {
  const { canceled, filePaths } = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'JSON Files', extensions: ['json'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });

  if (canceled) {
    return null;
  }

  try {
    const filePath = filePaths[0];
    const content = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(content);
  } catch (error) {
    console.error('Error reading file:', error);
    throw error;
  }
});

// Handle file saving
ipcMain.handle('saveFile', async (_, content) => {
  const { canceled, filePath } = await dialog.showSaveDialog(mainWindow, {
    filters: [
      { name: 'JSON Files', extensions: ['json'] },
      { name: 'All Files', extensions: ['*'] },
    ],
  });

  if (canceled) {
    return null;
  }

  try {
    await fs.writeFile(filePath, content, 'utf-8');
    return true;
  } catch (error) {
    console.error('Error saving file:', error);
    throw error;
  }
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