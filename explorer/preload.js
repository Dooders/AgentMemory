const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  openFiles: () => ipcRenderer.invoke('dialog:openFiles'),
  loadDefaultMemories: () => ipcRenderer.invoke('loadDefaultMemories')
}); 