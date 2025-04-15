import React, { useState } from 'react';
import styled from 'styled-components';

// Styled components
const EditorContainer = styled.div`
  flex: 1;
  background-color: ${({ theme }) => theme.colors.editor};
  display: flex;
  flex-direction: column;
  overflow: hidden;
`;

const TabBar = styled.div`
  display: flex;
  background-color: ${({ theme }) => theme.colors.editor};
  border-bottom: 1px solid ${({ theme }) => theme.colors.border};
  height: 36px;
`;

const Tab = styled.div<{ active: boolean }>`
  padding: 0 15px;
  height: 36px;
  display: flex;
  align-items: center;
  background-color: ${({ active, theme }) => 
    active ? theme.colors.background : theme.colors.editor};
  border-right: 1px solid ${({ theme }) => theme.colors.border};
  color: ${({ active, theme }) => 
    active ? theme.colors.text : 'rgba(255, 255, 255, 0.5)'};
  cursor: pointer;
  
  &:hover {
    background-color: ${({ active, theme }) => 
      active ? theme.colors.background : 'rgba(255, 255, 255, 0.05)'};
  }
`;

const ContentArea = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  padding: 20px;
  overflow: auto;
`;

// Visualization components
const MazeContainer = styled.div`
  margin-bottom: 20px;
  background-color: ${({ theme }) => theme.colors.sidebar};
  border-radius: 4px;
  padding: 20px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
`;

const MazeTitle = styled.h2`
  font-size: 18px;
  margin-top: 0;
  margin-bottom: 15px;
  color: ${({ theme }) => theme.colors.text};
`;

const MazeGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(10, 40px);
  grid-template-rows: repeat(10, 40px);
  gap: 2px;
`;

const MazeCell = styled.div<{ cellType: 'empty' | 'agent' | 'target' | 'obstacle' }>`
  width: 40px;
  height: 40px;
  background-color: ${({ cellType, theme }) => {
    switch (cellType) {
      case 'agent':
        return theme.colors.accent;
      case 'target':
        return '#00C853'; // Green
      case 'obstacle':
        return '#F44336'; // Red
      default:
        return theme.colors.sidebar;
    }
  }};
  border: 1px solid ${({ theme }) => theme.colors.border};
  display: flex;
  justify-content: center;
  align-items: center;
`;

const MetricsContainer = styled.div`
  display: flex;
  gap: 20px;
  margin-bottom: 20px;
`;

const MetricsPanel = styled.div`
  flex: 1;
  background-color: ${({ theme }) => theme.colors.sidebar};
  border-radius: 4px;
  padding: 20px;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
`;

const MetricsTitle = styled.h3`
  font-size: 16px;
  margin-top: 0;
  margin-bottom: 15px;
  color: ${({ theme }) => theme.colors.text};
`;

const ChartPlaceholder = styled.div`
  height: 200px;
  background-color: rgba(255, 255, 255, 0.05);
  border-radius: 4px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: ${({ theme }) => theme.colors.text};
`;

// Main component
const EditorArea: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('maze');
  
  // Mock maze data - in a real app, this would come from your simulation
  const mazeSize = 10;
  const agentPosition = [2, 3];
  const targetPosition = [8, 8];
  const obstacles = [[3, 3], [3, 4], [3, 5], [7, 7], [8, 7], [9, 7]];
  
  // Function to determine cell type
  const getCellType = (row: number, col: number): 'empty' | 'agent' | 'target' | 'obstacle' => {
    if (row === agentPosition[0] && col === agentPosition[1]) return 'agent';
    if (row === targetPosition[0] && col === targetPosition[1]) return 'target';
    if (obstacles.some(([r, c]) => r === row && c === col)) return 'obstacle';
    return 'empty';
  };
  
  return (
    <EditorContainer>
      <TabBar>
        <Tab active={activeTab === 'maze'} onClick={() => setActiveTab('maze')}>
          Maze Environment
        </Tab>
        <Tab active={activeTab === 'memory'} onClick={() => setActiveTab('memory')}>
          Memory Content
        </Tab>
      </TabBar>
      
      <ContentArea>
        {activeTab === 'maze' && (
          <>
            <MazeContainer>
              <MazeTitle>Maze Environment</MazeTitle>
              <MazeGrid>
                {Array.from({ length: mazeSize }).map((_, row) => 
                  Array.from({ length: mazeSize }).map((_, col) => (
                    <MazeCell 
                      key={`${row}-${col}`} 
                      cellType={getCellType(row, col)} 
                    />
                  ))
                )}
              </MazeGrid>
            </MazeContainer>
            
            <MetricsContainer>
              <MetricsPanel>
                <MetricsTitle>Rewards per Episode</MetricsTitle>
                <ChartPlaceholder>
                  Chart.js will be implemented here
                </ChartPlaceholder>
              </MetricsPanel>
              
              <MetricsPanel>
                <MetricsTitle>Steps per Episode</MetricsTitle>
                <ChartPlaceholder>
                  Chart.js will be implemented here
                </ChartPlaceholder>
              </MetricsPanel>
            </MetricsContainer>
          </>
        )}
        
        {activeTab === 'memory' && (
          <div>
            <h2>Memory Content</h2>
            <p>Memory content will be displayed here using Monaco Editor.</p>
          </div>
        )}
      </ContentArea>
    </EditorContainer>
  );
};

export default EditorArea; 