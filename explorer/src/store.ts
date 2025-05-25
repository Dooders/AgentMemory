import { create } from 'zustand';
import { Memory, FilterState, SortState } from './types';

interface AppState {
  memories: Memory[];
  filteredMemories: Memory[];
  selectedMemory: Memory | null;
  filters: FilterState;
  sort: SortState;
  setMemories: (memories: Memory[]) => void;
  setSelectedMemory: (memory: Memory | null) => void;
  setFilters: (filters: Partial<FilterState>) => void;
  setSort: (sort: Partial<SortState>) => void;
  applyFilters: () => void;
}

const initialFilters: FilterState = {
  dateRange: [null, null],
  roles: ['user', 'assistant', 'system'],
  searchText: '',
  minImportance: 0,
};

const initialSort: SortState = {
  field: 'timestamp',
  direction: 'desc',
};

export const useStore = create<AppState>((set, get) => ({
  memories: [],
  filteredMemories: [],
  selectedMemory: null,
  filters: initialFilters,
  sort: initialSort,

  setMemories: (memories) => {
    set({ memories, filteredMemories: memories });
  },

  setSelectedMemory: (memory) => {
    set({ selectedMemory: memory });
  },

  setFilters: (filters) => {
    set((state) => ({
      filters: { ...state.filters, ...filters },
    }));
    get().applyFilters();
  },

  setSort: (sort) => {
    set((state) => ({
      sort: { ...state.sort, ...sort },
    }));
    get().applyFilters();
  },

  applyFilters: () => {
    const { memories, filters, sort } = get();
    
    let filtered = [...memories];

    // Apply date range filter
    if (filters.dateRange[0] && filters.dateRange[1]) {
      filtered = filtered.filter((memory) => {
        const date = new Date(memory.timestamp);
        return date >= filters.dateRange[0]! && date <= filters.dateRange[1]!;
      });
    }

    // Apply role filter
    if (filters.roles.length > 0) {
      filtered = filtered.filter((memory) => filters.roles.includes(memory.role));
    }

    // Apply text search
    if (filters.searchText) {
      const searchLower = filters.searchText.toLowerCase();
      filtered = filtered.filter(
        (memory) =>
          memory.content.toLowerCase().includes(searchLower) ||
          memory.tags.some((tag) => tag.toLowerCase().includes(searchLower))
      );
    }

    // Apply importance filter
    filtered = filtered.filter((memory) => memory.importance >= filters.minImportance);

    // Apply sorting
    filtered.sort((a, b) => {
      const direction = sort.direction === 'asc' ? 1 : -1;
      if (sort.field === 'timestamp') {
        return direction * (new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
      }
      return direction * (a.importance - b.importance);
    });

    set({ filteredMemories: filtered });
  },
})); 