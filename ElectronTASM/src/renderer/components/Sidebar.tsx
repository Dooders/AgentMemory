import React from 'react';
import styled from 'styled-components';

// Types
interface SidebarProps {
  activeView: string;
}

// Styled components
const SidebarContainer = styled.div`
  width: ${({ theme }) => theme.sizes.sidebarWidth};
  height: 100%;
  background-color: ${({ theme }) => theme.colors.sidebar};
  border-right: 1px solid ${({ theme }) => theme.colors.border};
  overflow-y: auto;
`;

const SidebarHeader = styled.div`
  padding: 10px 20px;
  font-weight: bold;
  text-transform: uppercase;
  font-size: 11px;
  letter-spacing: 1px;
  color: ${({ theme }) => theme.colors.text};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const SidebarContent = styled.div`
  padding: 10px 0;
`;

const TreeItem = styled.div`
  padding: 5px 20px;
  cursor: pointer;
  display: flex;
  align-items: center;
  
  &:hover {
    background-color: rgba(255, 255, 255, 0.05);
  }
`;

const TreeItemIcon = styled.span`
  margin-right: 5px;
  font-size: 14px;
`;

const TreeItemLabel = styled.span`
  font-size: 13px;
`;

// Explorer content component
const ExplorerView: React.FC = () => (
  <>
    <SidebarHeader>
      Explorer
    </SidebarHeader>
    <SidebarContent>
      <TreeItem>
        <TreeItemIcon>📁</TreeItemIcon>
        <TreeItemLabel>src</TreeItemLabel>
      </TreeItem>
      <TreeItem>
        <TreeItemIcon>📁</TreeItemIcon>
        <TreeItemLabel>public</TreeItemLabel>
      </TreeItem>
      <TreeItem>
        <TreeItemIcon>📄</TreeItemIcon>
        <TreeItemLabel>package.json</TreeItemLabel>
      </TreeItem>
      <TreeItem>
        <TreeItemIcon>📄</TreeItemIcon>
        <TreeItemLabel>tsconfig.json</TreeItemLabel>
      </TreeItem>
    </SidebarContent>
  </>
);

// Memory browser component
const MemoryView: React.FC = () => (
  <>
    <SidebarHeader>
      Memory Browser
    </SidebarHeader>
    <SidebarContent>
      <TreeItem>
        <TreeItemIcon>🧠</TreeItemIcon>
        <TreeItemLabel>STM</TreeItemLabel>
      </TreeItem>
      <TreeItem>
        <TreeItemIcon>🧠</TreeItemIcon>
        <TreeItemLabel>ITM</TreeItemLabel>
      </TreeItem>
      <TreeItem>
        <TreeItemIcon>🧠</TreeItemIcon>
        <TreeItemLabel>LTM</TreeItemLabel>
      </TreeItem>
    </SidebarContent>
  </>
);

// Settings view component
const SettingsView: React.FC = () => (
  <>
    <SidebarHeader>
      Settings
    </SidebarHeader>
    <SidebarContent>
      <TreeItem>
        <TreeItemIcon>⚙️</TreeItemIcon>
        <TreeItemLabel>Memory Configuration</TreeItemLabel>
      </TreeItem>
      <TreeItem>
        <TreeItemIcon>⚙️</TreeItemIcon>
        <TreeItemLabel>Visualization Settings</TreeItemLabel>
      </TreeItem>
      <TreeItem>
        <TreeItemIcon>⚙️</TreeItemIcon>
        <TreeItemLabel>Agent Configuration</TreeItemLabel>
      </TreeItem>
    </SidebarContent>
  </>
);

// Main component
const Sidebar: React.FC<SidebarProps> = ({ activeView }) => {
  let content;
  
  switch (activeView) {
    case 'explorer':
      content = <ExplorerView />;
      break;
    case 'memory':
      content = <MemoryView />;
      break;
    case 'settings':
      content = <SettingsView />;
      break;
    default:
      content = <ExplorerView />;
  }
  
  return (
    <SidebarContainer>
      {content}
    </SidebarContainer>
  );
};

export default Sidebar; 