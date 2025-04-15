import React from 'react';
import styled from 'styled-components';

// Types
interface StatusBarProps {
  episode: number;
  step: number;
  reward: number;
  isRunning: boolean;
  onStep: () => void;
  onReset: () => void;
  onToggle: () => void;
}

// Styled components
const StatusBarContainer = styled.div`
  height: 32px;
  background-color: ${({ theme }) => theme.colors.statusBar};
  display: flex;
  align-items: center;
  padding: 0 20px;
  justify-content: space-between;
  color: white;
  font-size: 12px;
`;

const StatusItem = styled.div`
  display: flex;
  align-items: center;
  margin-right: 20px;
`;

const StatusLabel = styled.span`
  margin-right: 5px;
  font-weight: bold;
`;

const StatusValue = styled.span`
  background-color: rgba(255, 255, 255, 0.2);
  padding: 2px 8px;
  border-radius: 4px;
`;

const StatusItems = styled.div`
  display: flex;
`;

const ControlsContainer = styled.div`
  display: flex;
  align-items: center;
`;

const Button = styled.button`
  background-color: transparent;
  border: 1px solid rgba(255, 255, 255, 0.3);
  color: white;
  padding: 3px 10px;
  margin-left: 10px;
  border-radius: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
  
  &:hover {
    background-color: rgba(255, 255, 255, 0.1);
  }
  
  &:active {
    background-color: rgba(255, 255, 255, 0.2);
  }
`;

// Main component
const StatusBar: React.FC<StatusBarProps> = ({
  episode,
  step,
  reward,
  isRunning,
  onStep,
  onReset,
  onToggle
}) => {
  return (
    <StatusBarContainer>
      <StatusItems>
        <StatusItem>
          <StatusLabel>Episode:</StatusLabel>
          <StatusValue>{episode}</StatusValue>
        </StatusItem>
        
        <StatusItem>
          <StatusLabel>Step:</StatusLabel>
          <StatusValue>{step}</StatusValue>
        </StatusItem>
        
        <StatusItem>
          <StatusLabel>Reward:</StatusLabel>
          <StatusValue>{reward.toFixed(1)}</StatusValue>
        </StatusItem>
      </StatusItems>
      
      <ControlsContainer>
        <Button onClick={onStep}>Step</Button>
        <Button onClick={onToggle}>{isRunning ? 'Stop' : 'Start'}</Button>
        <Button onClick={onReset}>Reset</Button>
      </ControlsContainer>
    </StatusBarContainer>
  );
};

export default StatusBar; 