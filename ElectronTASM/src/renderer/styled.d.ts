import 'styled-components';

// Extend the DefaultTheme interface
declare module 'styled-components' {
  export interface DefaultTheme {
    colors: {
      background: string;
      activityBar: string;
      sidebar: string;
      editor: string;
      statusBar: string;
      text: string;
      accent: string;
      border: string;
      activityBarIconActive: string;
      activityBarIconInactive: string;
    };
    fonts: {
      main: string;
      code: string;
    };
    sizes: {
      activityBarWidth: string;
      sidebarWidth: string;
      statusBarHeight: string;
    };
  }
} 