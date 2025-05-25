import React from 'react';
import { Box, CssBaseline, ThemeProvider, createTheme } from '@mui/material';
import { LocalizationProvider } from '@mui/x-date-pickers';
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import TopBar from './components/TopBar';
import SideBar from './components/SideBar';
import MainPanel from './components/MainPanel';
import FilterDrawer from './components/FilterDrawer';
import StatusBar from './components/StatusBar';

const theme = createTheme({
  palette: {
    mode: 'dark',
  },
});

const App: React.FC = () => {
  return (
    <ThemeProvider theme={theme}>
      <LocalizationProvider dateAdapter={AdapterDateFns}>
        <CssBaseline />
        <Box sx={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
          <TopBar />
          <Box sx={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
            <SideBar />
            <MainPanel />
            <FilterDrawer />
          </Box>
          <StatusBar />
        </Box>
      </LocalizationProvider>
    </ThemeProvider>
  );
};

export default App; 