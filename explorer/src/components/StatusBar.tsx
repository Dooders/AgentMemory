import React from 'react';
import { Box, Paper, Typography, Button } from '@mui/material';
import { Save as SaveIcon } from '@mui/icons-material';
import { useStore } from '../store';

const StatusBar: React.FC = () => {
  const { memories, filteredMemories } = useStore();

  const handleExport = async () => {
    try {
      await window.electron.saveFile(JSON.stringify(filteredMemories, null, 2));
    } catch (error) {
      console.error('Error exporting memories:', error);
    }
  };

  return (
    <Paper
      sx={{
        p: 1,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}
    >
      <Typography variant="body2" color="text.secondary">
        {filteredMemories.length} / {memories.length} memories loaded
        {filteredMemories.length !== memories.length
          ? ` • ${memories.length - filteredMemories.length} filtered`
          : ''}
      </Typography>
      <Button
        startIcon={<SaveIcon />}
        size="small"
        onClick={handleExport}
        disabled={filteredMemories.length === 0}
      >
        Export
      </Button>
    </Paper>
  );
};

export default StatusBar; 