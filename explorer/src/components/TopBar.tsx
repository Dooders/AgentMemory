import React from 'react';
import {
  AppBar,
  Toolbar,
  IconButton,
  TextField,
  Box,
  Typography,
} from '@mui/material';
import { FolderOpen as FolderOpenIcon } from '@mui/icons-material';
import { useStore } from '../store';

const TopBar: React.FC = () => {
  const { filters, setFilters } = useStore();

  const handleOpenFile = async () => {
    try {
      const result = await window.electron.openFile();
      if (result) {
        // Handle the opened file data
        console.log('File opened:', result);
      }
    } catch (error) {
      console.error('Error opening file:', error);
    }
  };

  const handleSearchChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    setFilters({ searchText: event.target.value });
  };

  return (
    <AppBar position="static" color="default" elevation={1}>
      <Toolbar>
        <IconButton
          edge="start"
          color="inherit"
          aria-label="open file"
          onClick={handleOpenFile}
        >
          <FolderOpenIcon />
        </IconButton>
        <Typography variant="h6" sx={{ ml: 2, mr: 4 }}>
          Agent Memory Explorer
        </Typography>
        <Box sx={{ flexGrow: 1 }}>
          <TextField
            fullWidth
            variant="outlined"
            size="small"
            placeholder="Search memories..."
            value={filters.searchText}
            onChange={handleSearchChange}
          />
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default TopBar; 