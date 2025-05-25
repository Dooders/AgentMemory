export interface Memory {
  id: string;
  timestamp: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  importance: number;
  embedding?: number[];
  tags: string[];
}

export interface FilterState {
  dateRange: [Date | null, Date | null];
  roles: ('user' | 'assistant' | 'system')[];
  searchText: string;
  minImportance: number;
}

export interface SortState {
  field: 'timestamp' | 'importance';
  direction: 'asc' | 'desc';
} 