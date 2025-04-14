// Preload script for contextBridge API
import { contextBridge, ipcRenderer } from 'electron';

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld(
  'electronAPI', {
    memorySystem: {
      initialize: (config: any) => ipcRenderer.invoke('memory-system:initialize', config),
      getMemories: (params: { tier: string, agentId: string }) => 
        ipcRenderer.invoke('memory-system:get-memories', params),
    },
    agent: {
      step: (action: any) => ipcRenderer.invoke('agent:step', action),
      reset: () => ipcRenderer.invoke('agent:reset'),
    },
    app: {
      getVersion: () => ipcRenderer.invoke('app:get-version'),
    }
  }
); 