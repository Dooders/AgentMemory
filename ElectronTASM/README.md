# TASM Visualizer

A VS Code-like interface for visualizing the TASM (Tiered Agent-based Semantic Memory) system, built with Electron, React, and TypeScript.

## Overview

This application reimplements the TASM Python-based visualizer as a modern cross-platform desktop application. It provides a better user experience with a familiar VS Code-like interface, making it easier to visualize and debug memory-enhanced agents.

## Features

- VS Code-like user interface
- Maze environment visualization
- Memory tier display (STM, ITM, LTM)
- Performance metrics visualization
- Simulation controls (Step, Start/Stop, Reset)
- Dark theme matching VS Code
- Activity bar (left sidebar with icons)
- Side panel for file explorer/memory browser
- Editor area for main visualization

## Development

### Prerequisites

- Node.js (v14+)
- npm or yarn

### Setup

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```
3. Start the development server:
   ```
   npm run dev
   ```

### Build

To build the application for distribution:

```
npm run build
```

## Architecture

- **Electron**: Application framework with main and renderer processes
- **React**: UI components
- **TypeScript**: Type-safe JavaScript
- **Styled Components**: VS Code-like styling
- **Monaco Editor**: For code editors (VS Code's editor)
- **Chart.js**: For performance metrics
- **Python Bridge**: For connecting to the memory system backend

## Project Structure

```
ElectronTASM/
├── assets/          # Application assets
├── src/
│   ├── main/        # Electron main process
│   │   ├── main.ts  # Main entry point
│   │   └── preload.ts # Preload script
│   └── renderer/    # React renderer
│       ├── components/ # UI components
│       ├── App.tsx  # Main React component
│       └── index.tsx # React entry point
├── package.json
└── tsconfig.json
``` 