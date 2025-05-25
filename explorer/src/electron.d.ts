interface ElectronAPI {
  openFile: () => Promise<any>;
  saveFile: (content: string) => Promise<boolean>;
}

declare global {
  interface Window {
    electron: ElectronAPI;
  }
} 