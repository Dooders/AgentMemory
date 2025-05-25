const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  openFile: () => ipcRenderer.invoke('openFile'),
  saveFile: (content) => ipcRenderer.invoke('saveFile', content),
});

contextBridge.exposeInMainWorld('electronAPI', {
  openFiles: () => ipcRenderer.invoke('dialog:openFiles'),
  loadDefaultMemories: () => ipcRenderer.invoke('loadDefaultMemories')
}); 