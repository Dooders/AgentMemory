import React from 'react';
import { Box, Paper, Typography } from '@mui/material';
import { FixedSizeList as List } from 'react-window';
import { useStore } from '../store';
import { format } from 'date-fns';
import { Memory } from '../types';

const MemoryItem: React.FC<{
  index: number;
  style: React.CSSProperties;
  data: Memory[];
}> = ({ index, style, data }) => {
  const memory = data[index];
  const { setSelectedMemory } = useStore();

  const getRoleColor = (role: string) => {
    switch (role) {
      case 'user':
        return '#2196f3';
      case 'assistant':
        return '#4caf50';
      case 'system':
        return '#f44336';
      default:
        return '#757575';
    }
  };

  return (
    <Box
      style={style}
      sx={{
        display: 'flex',
        alignItems: 'center',
        padding: '8px 16px',
        cursor: 'pointer',
        '&:hover': {
          backgroundColor: 'rgba(255, 255, 255, 0.08)',
        },
      }}
      onClick={() => setSelectedMemory(memory)}
    >
      <Box
        sx={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          backgroundColor: getRoleColor(memory.role),
          marginRight: 2,
        }}
      />
      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Typography
          variant="body2"
          sx={{
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis',
          }}
        >
          {memory.content.substring(0, 50)}
          {memory.content.length > 50 ? '...' : ''}
        </Typography>
        <Typography variant="caption" color="text.secondary">
          {format(new Date(memory.timestamp), 'MMM d, yyyy HH:mm')}
        </Typography>
      </Box>
    </Box>
  );
};

const SideBar: React.FC = () => {
  const { filteredMemories } = useStore();

  return (
    <Paper
      sx={{
        width: 400,
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
      }}
    >
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Typography variant="h6">Memories</Typography>
      </Box>
      <Box sx={{ flex: 1, overflow: 'hidden' }}>
        <List
          height={window.innerHeight - 120}
          width="100%"
          itemCount={filteredMemories.length}
          itemSize={72}
          itemData={filteredMemories}
        >
          {MemoryItem}
        </List>
      </Box>
    </Paper>
  );
};

export default SideBar; 