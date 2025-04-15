import React, { useState, useEffect } from 'react';
import styled, { createGlobalStyle, ThemeProvider } from 'styled-components';

// Import components
import ActivityBar from './components/ActivityBar';
import Sidebar from './components/Sidebar';
import EditorArea from './components/EditorArea';
import StatusBar from './components/StatusBar';

// Define VS Code-like theme
const vscodeTheme = {
  colors: {
    background: '#1e1e1e',
    activityBar: '#333333',
    sidebar: '#252526',
    editor: '#1e1e1e',
    statusBar: '#007acc',
    text: '#cccccc',
    accent: '#007acc',
    border: '#474747',
    activityBarIconActive: '#ffffff',
    activityBarIconInactive: '#858585',
  },
  fonts: {
    main: '"Segoe UI", Tahoma, Geneva, Verdana, sans-serif',
    code: '"Consolas", "Courier New", monospace',
  },
  sizes: {
    activityBarWidth: '48px',
    sidebarWidth: '300px',
    statusBarHeight: '22px',
  },
};

// Global styles
const GlobalStyle = createGlobalStyle`
  body {
    margin: 0;
    padding: 0;
    font-family: ${({ theme }) => theme.fonts.main};
    background-color: ${({ theme }) => theme.colors.background};
    color: ${({ theme }) => theme.colors.text};
    overflow: hidden;
  }
  * {
    box-sizing: border-box;
  }
`;

// Styled components
const AppContainer = styled.div`
  display: flex;
  flex-direction: column;
  height: 100vh;
  width: 100vw;
`;

const MainArea = styled.div`
  display: flex;
  flex: 1;
  overflow: hidden;
`;

// Type declarations for the Window API
declare global {
  interface Window {
    electronAPI?: {
      memorySystem: {
        initialize: (config: any) => Promise<any>;
        getMemories: (params: { tier: string, agentId: string }) => Promise<any>;
      };
      agent: {
        step: (action: any) => Promise<any>;
        reset: () => Promise<any>;
      };
      app: {
        getVersion: () => Promise<string>;
      };
    };
  }
}

const App: React.FC = () => {
  // State for the active view in the activity bar
  const [activeView, setActiveView] = useState<string>('explorer');
  
  // State for the simulation
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [currentStep, setCurrentStep] = useState<number>(0);
  const [episode, setEpisode] = useState<number>(0);
  const [reward, setReward] = useState<number>(0);
  
  // State for the maze environment
  const [mazeData, setMazeData] = useState<any>(null);

  // Initialize the application
  useEffect(() => {
    // Initialize memory system if electronAPI is available
    if (window.electronAPI) {
      window.electronAPI.memorySystem.initialize({
        // Configuration for memory system
      }).then(result => {
        console.log('Memory system initialized:', result);
      }).catch(error => {
        console.error('Failed to initialize memory system:', error);
      });
    }
  }, []);

  // Placeholder for step function
  const handleStep = async () => {
    if (window.electronAPI) {
      try {
        const result = await window.electronAPI.agent.step({ direction: 0 });
        setCurrentStep(prev => prev + 1);
        setReward(prev => prev + result.reward);
        // Update maze data with result.observation
      } catch (error) {
        console.error('Step failed:', error);
      }
    }
  };

  // Placeholder for reset function
  const handleReset = async () => {
    if (window.electronAPI) {
      try {
        await window.electronAPI.agent.reset();
        setCurrentStep(0);
        setEpisode(prev => prev + 1);
        setReward(0);
        // Reset maze data
      } catch (error) {
        console.error('Reset failed:', error);
      }
    }
  };

  // Placeholder for toggle simulation
  const toggleSimulation = () => {
    setIsRunning(!isRunning);
    // Logic for running simulation steps
  };

  return (
    <ThemeProvider theme={vscodeTheme}>
      <GlobalStyle />
      <AppContainer>
        <MainArea>
          <ActivityBar activeView={activeView} setActiveView={setActiveView} />
          <Sidebar activeView={activeView} />
          <EditorArea />
        </MainArea>
        <StatusBar 
          episode={episode} 
          step={currentStep} 
          reward={reward} 
          isRunning={isRunning}
          onStep={handleStep}
          onReset={handleReset}
          onToggle={toggleSimulation}
        />
      </AppContainer>
    </ThemeProvider>
  );
};

export default App; 