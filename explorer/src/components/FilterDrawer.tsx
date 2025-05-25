import React from 'react';
import {
  Drawer,
  Box,
  Typography,
  FormGroup,
  FormControlLabel,
  Checkbox,
  Slider,
  Divider,
  IconButton,
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers';
import { Close as CloseIcon } from '@mui/icons-material';
import { useStore } from '../store';

const FilterDrawer: React.FC = () => {
  const { filters, setFilters, sort, setSort } = useStore();
  const [open, setOpen] = React.useState(true);

  const handleRoleChange = (role: string) => (event: React.ChangeEvent<HTMLInputElement>) => {
    const newRoles = event.target.checked
      ? [...filters.roles, role as 'user' | 'assistant' | 'system']
      : filters.roles.filter((r) => r !== role);
    setFilters({ roles: newRoles });
  };

  const handleImportanceChange = (_: Event, value: number | number[]) => {
    setFilters({ minImportance: value as number });
  };

  const handleSortChange = (field: 'timestamp' | 'importance') => () => {
    setSort({
      field,
      direction: sort.field === field && sort.direction === 'asc' ? 'desc' : 'asc',
    });
  };

  return (
    <Drawer
      variant="persistent"
      anchor="right"
      open={open}
      sx={{
        width: 300,
        flexShrink: 0,
        '& .MuiDrawer-paper': {
          width: 300,
          boxSizing: 'border-box',
        },
      }}
    >
      <Box sx={{ p: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h6">Filters</Typography>
        <IconButton onClick={() => setOpen(false)}>
          <CloseIcon />
        </IconButton>
      </Box>
      <Divider />
      <Box sx={{ p: 2 }}>
        <Typography variant="subtitle1" gutterBottom>
          Date Range
        </Typography>
        <Box sx={{ display: 'flex', gap: 1, mb: 3 }}>
          <DatePicker
            label="Start"
            value={filters.dateRange[0]}
            onChange={(date) => setFilters({ dateRange: [date, filters.dateRange[1]] })}
            slotProps={{ textField: { size: 'small', fullWidth: true } }}
          />
          <DatePicker
            label="End"
            value={filters.dateRange[1]}
            onChange={(date) => setFilters({ dateRange: [filters.dateRange[0], date] })}
            slotProps={{ textField: { size: 'small', fullWidth: true } }}
          />
        </Box>

        <Typography variant="subtitle1" gutterBottom>
          Roles
        </Typography>
        <FormGroup sx={{ mb: 3 }}>
          <FormControlLabel
            control={
              <Checkbox
                checked={filters.roles.includes('user')}
                onChange={handleRoleChange('user')}
              />
            }
            label="User"
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={filters.roles.includes('assistant')}
                onChange={handleRoleChange('assistant')}
              />
            }
            label="Assistant"
          />
          <FormControlLabel
            control={
              <Checkbox
                checked={filters.roles.includes('system')}
                onChange={handleRoleChange('system')}
              />
            }
            label="System"
          />
        </FormGroup>

        <Typography variant="subtitle1" gutterBottom>
          Minimum Importance
        </Typography>
        <Slider
          value={filters.minImportance}
          onChange={handleImportanceChange}
          min={0}
          max={1}
          step={0.01}
          valueLabelDisplay="auto"
          sx={{ mb: 3 }}
        />

        <Typography variant="subtitle1" gutterBottom>
          Sort By
        </Typography>
        <Box sx={{ display: 'flex', gap: 1 }}>
          <Box
            sx={{
              flex: 1,
              p: 1,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1,
              cursor: 'pointer',
              bgcolor: sort.field === 'timestamp' ? 'action.selected' : 'transparent',
            }}
            onClick={handleSortChange('timestamp')}
          >
            <Typography variant="body2" align="center">
              Timestamp {sort.field === 'timestamp' && (sort.direction === 'asc' ? '↑' : '↓')}
            </Typography>
          </Box>
          <Box
            sx={{
              flex: 1,
              p: 1,
              border: 1,
              borderColor: 'divider',
              borderRadius: 1,
              cursor: 'pointer',
              bgcolor: sort.field === 'importance' ? 'action.selected' : 'transparent',
            }}
            onClick={handleSortChange('importance')}
          >
            <Typography variant="body2" align="center">
              Importance {sort.field === 'importance' && (sort.direction === 'asc' ? '↑' : '↓')}
            </Typography>
          </Box>
        </Box>
      </Box>
    </Drawer>
  );
};

export default FilterDrawer; 