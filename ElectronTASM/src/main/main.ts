import { app, BrowserWindow, ipcMain } from 'electron';
import * as path from 'path';
import * as isDev from 'electron-is-dev';
import Store from 'electron-store';

// Initialize store for app settings
const store = new Store();

let mainWindow: BrowserWindow | null = null;

function createWindow() {
  // Create the browser window.
  mainWindow = new BrowserWindow({
    width: 1920,
    height: 1080,
    minWidth: 1280,
    minHeight: 720,
    backgroundColor: '#1a1a1a',
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js'),
    },
    show: false,
  });

  // Load the index.html of the app.
  // Instead of using a development server, load directly from the file system
  const startUrl = `file://${path.join(__dirname, '../renderer/index.html')}`;
  mainWindow.loadURL(startUrl);

  // Open the DevTools in development mode
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // Show window when ready to avoid flickering
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show();
  });

  // Emitted when the window is closed.
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
app.on('ready', createWindow);

// Quit when all windows are closed.
app.on('window-all-closed', () => {
  // On macOS it is common for applications to stay open until the user quits
  // explicitly with Cmd + Q
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  // On macOS it's common to re-create a window when the dock icon is clicked
  if (mainWindow === null) {
    createWindow();
  }
});

// In this file you can include the rest of your app's specific main process
// code. You can also put them in separate files and import them here.

// IPC handlers for communication with the Python backend
ipcMain.handle('memory-system:initialize', async (event, config) => {
  // Here you would initialize the memory system, potentially spawning a Python process
  console.log('Initializing memory system with config:', config);
  // Return mock data for now
  return { success: true, message: 'Memory system initialized' };
});

ipcMain.handle('memory-system:get-memories', async (event, { tier, agentId }) => {
  // Here you would request memories from the specific tier
  console.log(`Getting ${tier} memories for agent ${agentId}`);
  // Return mock data for now
  return { 
    memories: [
      { id: 1, content: 'Memory 1 content', timestamp: Date.now() },
      { id: 2, content: 'Memory 2 content', timestamp: Date.now() - 1000 }
    ] 
  };
});

ipcMain.handle('agent:step', async (event, action) => {
  // Here you would execute a step in the simulation
  console.log('Executing step with action:', action);
  // Return mock data for now
  return { 
    observation: { position: [3, 4], target: [8, 8] },
    reward: 0.5,
    done: false
  };
}); 