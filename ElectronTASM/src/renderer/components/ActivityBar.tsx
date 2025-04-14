import React from 'react';
import styled from 'styled-components';

// Types
interface ActivityBarProps {
  activeView: string;
  setActiveView: (view: string) => void;
}

// Styled components
const ActivityBarContainer = styled.div`
  width: ${({ theme }) => theme.sizes.activityBarWidth};
  height: 100%;
  background-color: ${({ theme }) => theme.colors.activityBar};
  display: flex;
  flex-direction: column;
  align-items: center;
  padding-top: 10px;
`;

const ActivityBarIcon = styled.div<{ active: boolean }>`
  width: 36px;
  height: 36px;
  display: flex;
  justify-content: center;
  align-items: center;
  margin-bottom: 10px;
  cursor: pointer;
  color: ${({ active, theme }) => 
    active ? theme.colors.activityBarIconActive : theme.colors.activityBarIconInactive};
  border-left: ${({ active, theme }) => 
    active ? `2px solid ${theme.colors.accent}` : '2px solid transparent'};
  
  &:hover {
    color: ${({ theme }) => theme.colors.activityBarIconActive};
  }
`;

// Mock icons until we have proper SVG icons
const ExplorerIcon = () => <span style={{ fontSize: '20px' }}>📁</span>;
const MemoryIcon = () => <span style={{ fontSize: '20px' }}>🧠</span>;
const SettingsIcon = () => <span style={{ fontSize: '20px' }}>⚙️</span>;

// Main component
const ActivityBar: React.FC<ActivityBarProps> = ({ activeView, setActiveView }) => {
  return (
    <ActivityBarContainer>
      <ActivityBarIcon 
        active={activeView === 'explorer'} 
        onClick={() => setActiveView('explorer')}
        title="Explorer"
      >
        <ExplorerIcon />
      </ActivityBarIcon>
      
      <ActivityBarIcon 
        active={activeView === 'memory'} 
        onClick={() => setActiveView('memory')}
        title="Memory Browser"
      >
        <MemoryIcon />
      </ActivityBarIcon>
      
      <ActivityBarIcon 
        active={activeView === 'settings'} 
        onClick={() => setActiveView('settings')}
        title="Settings"
      >
        <SettingsIcon />
      </ActivityBarIcon>
    </ActivityBarContainer>
  );
};

export default ActivityBar; 