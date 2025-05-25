import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Chip,
  IconButton,
  Collapse,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  ExpandLess as ExpandLessIcon,
} from '@mui/icons-material';
import { useStore } from '../store';
import { format } from 'date-fns';

const MainPanel: React.FC = () => {
  const { selectedMemory } = useStore();
  const [showEmbedding, setShowEmbedding] = React.useState(false);

  if (!selectedMemory) {
    return (
      <Box
        sx={{
          flex: 1,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          p: 3,
        }}
      >
        <Typography color="text.secondary">
          Select a memory to view its details
        </Typography>
      </Box>
    );
  }

  return (
    <Paper
      sx={{
        flex: 1,
        height: '100%',
        overflow: 'auto',
        p: 3,
      }}
    >
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          Memory Details
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          ID: {selectedMemory.id}
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Timestamp: {format(new Date(selectedMemory.timestamp), 'PPpp')}
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Role: {selectedMemory.role}
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Importance: {selectedMemory.importance.toFixed(2)}
        </Typography>
      </Box>

      <Box sx={{ mb: 3 }}>
        <Typography variant="subtitle1" gutterBottom>
          Content
        </Typography>
        <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
          {selectedMemory.content}
        </Typography>
      </Box>

      {selectedMemory.tags.length > 0 && (
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            Tags
          </Typography>
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
            {selectedMemory.tags.map((tag) => (
              <Chip key={tag} label={tag} size="small" />
            ))}
          </Box>
        </Box>
      )}

      {selectedMemory.embedding && (
        <Box>
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              mb: 1,
            }}
          >
            <Typography variant="subtitle1">Embedding</Typography>
            <IconButton
              size="small"
              onClick={() => setShowEmbedding(!showEmbedding)}
            >
              {showEmbedding ? <ExpandLessIcon /> : <ExpandMoreIcon />}
            </IconButton>
          </Box>
          <Collapse in={showEmbedding}>
            <Typography
              variant="body2"
              sx={{
                fontFamily: 'monospace',
                whiteSpace: 'pre-wrap',
                backgroundColor: 'rgba(0, 0, 0, 0.1)',
                p: 1,
                borderRadius: 1,
              }}
            >
              {JSON.stringify(selectedMemory.embedding, null, 2)}
            </Typography>
          </Collapse>
        </Box>
      )}
    </Paper>
  );
};

export default MainPanel; 